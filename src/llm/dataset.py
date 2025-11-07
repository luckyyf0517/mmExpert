import os
import json
from tkinter import FALSE
import torch
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import copy
import transformers
from easydict import EasyDict as edict
from torch.utils.data import Dataset, ConcatDataset
import random
import yaml
import os.path as osp
from src.logger import log_message

import sys; sys.path.append('.')
from src.llm.utils import conversation as conversation_lib
from src.clip.dataset import load_radar_data

# Constants
IGNORE_INDEX = -100
DEFAULT_WAVE_INDICATOR = "<wave>"
DEFAULT_WAVE_PATCH_TOKEN = "<wave_patch>"
DEFAULT_WAVE_START_TOKEN = "<wave_bos>"
DEFAULT_WAVE_END_TOKEN = "<wave_eos>"
DEFAULT_WAVE_TOKEN_LEN = 248

# Default configuration for radar dataset
DEFAULT_REAL_CONFIG = {
    'max_motion_length': 496,
    'min_motion_len': 96,
    'unit_length': 16,
    'raw': True,
    'thresholding': True,
    'normalize': 'per_frame',  # 'none', 'per_frame', 'global', or 'log'
}

# Default question prompts for radar data
DEFAULT_QUESTION_PROMPTS = [
    "Describe the human motion based on the provided radar signals including range, Doppler, and azimuth information.",
    "Interpret the radar data and explain the corresponding human motion patterns.",
    "Analyze the given radar signals to describe the overall movement of the human body.",
    "Given radar data representing motion patterns, explain how a human is moving.",
    "Use the provided radar signals to describe the sequence of human body movements.",
    "Interpret the radar data to describe the motion pattern of the human body.",
    "Based on the radar signal data, explain the human motion being represented.",
    "Analyze the radar signals to generate a description of human motion.",
    "Use the radar data to interpret and describe the human body's movement.",
    "Given the radar signals, describe the motion of the human body in detail."
]


# NOTE: The following functions have been removed as they were unused:
# - DataCollatorForWaveTextDataset: Collation is handled by datamodule.py
# - preprocess_multimodal_wave: Preprocessing is handled by datamodule.py
# - preprocess: Label masking logic is in datamodule.py (with correct +1 fix)
# - make_object_wave_data_module: DataModule pattern is used instead


class WaveCaptionDataset(Dataset):
    """Dataset for wave caption tasks with multimodal support."""
    
    def __init__(self, data_root=None, split="train", tokenizer=None) -> None:
        super(WaveCaptionDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.tokenizer = tokenizer
        self.opt = self._load_config(data_root, split)
        self.data = self._load_data()
        
        log_message("LOAD", f"load {len(self.data)} data as {self.split} set", color="green")
    
    def _load_config(self, data_root, split):
        """Load configuration based on split type."""
        if split == 'real':
            return edict(DEFAULT_REAL_CONFIG)
        else:
            config_path = osp.join(data_root, 'radar_encoder_config.yaml')
            try:
                with open(config_path, 'r') as file:
                    yaml_data = yaml.safe_load(file)
            except yaml.constructor.ConstructorError:
                from yaml import UnsafeLoader
                with open(config_path, 'r') as file:
                    yaml_data = yaml.load(file, Loader=UnsafeLoader)
            
            if yaml_data is None:
                raise ValueError(f"YAML config is empty: {config_path}")
            
            yaml_data = edict(yaml_data)
            
            # Try to get dataset_cfg, fallback to default if not found
            if 'dataset_cfg' in yaml_data and yaml_data.dataset_cfg is not None:
                return edict(yaml_data.dataset_cfg)
            else:
                # If dataset_cfg is not in YAML, return default config
                return edict(DEFAULT_REAL_CONFIG)
    
    def _get_radar_path(self, data_item):
        """Get radar NPZ file path using the same logic as base_dataset.py."""
        motion_folder = data_item['filefolder']
        # Replace udoppler with mmwave (matching base_dataset.py)
        mmwave_postfix = self.opt.get('mmwave_postfix', '')
        motion_folder = motion_folder.replace('udoppler', 'mmwave' + mmwave_postfix)
        motion_index = data_item['fileindex']
        
        # Return NPZ file path (radar format from DATA_FORMAT.md)
        radar_path = os.path.join(motion_folder, f'{motion_index}.npz')
        
        if not os.path.exists(radar_path):
            raise FileNotFoundError(f"Radar file not found: {radar_path}")
        return radar_path
    
    def _load_data(self):
        """Load and process dataset."""
        if self.split == 'real':
            data_path = 'dataset/REAL/all.json'
        else:
            # Load data from _split directory
            data_path = osp.join(self.data_root, '_split', f'{self.split}.json')
        
        with open(data_path) as f:
            data = json.load(f)
        
        processed_data = []
        for i in data:
            if 'captions' not in data[i]:
                data[i]['captions'] = [data[i]['classname']]
        
            question_qas = self._generate_question_qas(data[i])
            caption_qas = self._generate_caption_qas(data[i])
            
            if 'train' in self.split:
                # assign caption QA to each data item
                data_items_caption = self._create_data_items(data[i], {'question': None, 'answer': ''})
                for data_item in data_items_caption:
                    qa = random.choice(caption_qas)
                    data_item['question'] = qa['question']
                    data_item['answer'] = qa['answer']
                processed_data.extend(data_items_caption)
                # assign question QA to each data item
                data_items = self._create_data_items(data[i], {'question': None, 'answer': ''})
                for data_item in data_items:
                    qa = random.choice(question_qas)
                    data_item['question'] = qa['question']
                    data_item['answer'] = qa['answer']
                processed_data.extend(data_items)
            elif 'test' in self.split:
                if self.opt.get('caption_only'):
                    qa_caption = caption_qas[0]
                    processed_data.extend(self._create_data_items(data[i], qa_caption))
                else: 
                    qa_question = random.choice(question_qas)
                    # qa_question includes a caption related QA
                    # processed_data.extend(self._create_data_items(data[i], qa_caption))
                    processed_data.extend(self._create_data_items(data[i], qa_question))
            else: 
                raise ValueError(f"Invalid split: {self.split}")
        
        return processed_data
    
    def _generate_caption_qas(self, data_item):
        """Generate caption-based QA pair."""
        if 'train' in self.split:
            qas = []
            for caption in data_item['captions']:
                qas.append({'question': None, 'answer': caption})
            return qas
        elif 'test' in self.split: 
            return [{'question': None, 'answer': '#'.join(data_item['captions'])}]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
    def _generate_question_qas(self, data_item):
        """Generate question-based QA pairs."""
        questions = list(data_item['questions'].values()) if 'questions' in data_item else []
        
        if 'test' in self.split and questions:
            return [random.choice(questions)]
        else:
            return questions
    
    def _create_data_items(self, data_item, qa):
        """Create data items using NPZ file format (new data organization)."""
        # New data format uses single NPZ file per sample (no rotation postfixes)
        radar_path = self._get_radar_path(data_item)
        
        data_items = [{
            'filename': radar_path,
            'question': qa.get('question') if qa else None,
            'answer': qa.get('answer') if qa else '',
            'caption': "\n".join(data_item['captions'])
        }]
        
        return data_items

    def __len__(self):
        """Return number of utterances."""
        return len(self.data)
    
    @staticmethod
    def default_question():
        """Get a random default question from predefined prompts."""
        return random.choice(DEFAULT_QUESTION_PROMPTS)
        
    @staticmethod
    def format_caption(question, answer):
        """Format question and answer into conversation format."""
        return [{
            "from": "human",
            "value": f"{DEFAULT_WAVE_INDICATOR}\n{question}"
        }, {
            "from": "gpt",
            "value": answer
        }]
    
    def __getitem__(self, index):
        """Get raw data item without preprocessing.
        
        Preprocessing is handled by datamodule.py's ConversationDatasetWrapper.
        This method only loads and returns raw data: question, answer, wave_embed.
        """
        instance = deepcopy(self.data[index])
        if instance['question'] is None:
            instance['question'] = self.default_question()

        # Load radar data from NPZ file using the same pipeline as base_dataset
        radar_data = load_radar_data(instance['filename'], self.opt)

        # Convert to tensors
        wave_embed = {
            'range_time': torch.tensor(radar_data['range_time']).float(),
            'doppler_time': torch.tensor(radar_data['doppler_time']).float(),
            'azimuth_time': torch.tensor(radar_data['azimuth_time']).float()
        }

        # Return raw data only - preprocessing done by datamodule.py
        return {
            'question': instance['question'],
            'answer': instance['answer'],
            'caption': instance['caption'],
            'wave_embed': wave_embed
        }
    

# NOTE: mini_dataset() function has been removed as it was unused and called deleted preprocess().


def init_tokenizer_only(model_path):
    """Initialize only tokenizer without loading the full model."""
    from transformers import AutoTokenizer
    
    # Set cache directory for offline mode
    cache_dir = "/root/autodl-tmp/mmExpert/huggingface"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=True  # Force offline mode
    )
    
    return tokenizer

if __name__ == "__main__":
    import sys
    import os
    
    # Import conversation modules
    from src.llm.utils import conversation as conversation_lib
    from src.llm.utils.conversation import conv_templates
    
    # Set test parameters
    data_root = "feature/clip_v1.11_bs64_r8a16"
    model_name = "huggingface/models--microsoft--Phi-3-mini-4k-instruct/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85"
    
    # Only get tokenizer, not the full model
    tokenizer = init_tokenizer_only(model_name)
    
    # Set the conversation template like in eval_llm
    conversation_lib.default_conversation = conversation_lib.conv_templates["conv_phi3"]
    conv = conv_templates["conv_phi3"].copy()
    
    # # Test train mode
    # print("=== TRAIN MODE ===")
    # train_dataset = WaveCaptionDataset(data_root=data_root, split="train", tokenizer=tokenizer)
    # if len(train_dataset) > 0:
    #     train_item = train_dataset[0]
    #     print(f"Train item keys: {list(train_item.keys())}")
    #     print(f"Question: {train_item['question']}")
    #     print(f"Answer: {train_item['answer'][:100]}...")
    #     print(f"Wave embed shape: {train_item['wave_embed'].shape}")
    
    # Test test mode
    log_message("DEBUG", "=== TEST MODE ===", color="cyan")
    test_dataset = WaveCaptionDataset(data_root=data_root, split="test_QAs", tokenizer=tokenizer)

    test_item = test_dataset[0]
    log_message("DEBUG", "---", color="cyan")
    log_message("DEBUG", f"Question: {test_item['question']}", color="cyan")
    log_message("DEBUG", f"Answer: {test_item['answer']}", color="cyan")

    test_item = test_dataset[1]
    log_message("DEBUG", "---", color="cyan")
    log_message("DEBUG", f"Question: {test_item['question']}", color="cyan")
    log_message("DEBUG", f"Answer: {test_item['answer']}", color="cyan")

    test_item = test_dataset[2]
    log_message("DEBUG", "---", color="cyan")
    log_message("DEBUG", f"Question: {test_item['question']}", color="cyan")
    log_message("DEBUG", f"Answer: {test_item['answer']}", color="cyan")
    