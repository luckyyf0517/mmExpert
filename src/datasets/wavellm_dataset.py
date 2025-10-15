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

import sys; sys.path.append('.')
from src.wavellm import conversation as conversation_lib
from src.datasets.base_dataset import process_motion

# Constants
IGNORE_INDEX = -100
DEFAULT_WAVE_INDICATOR = "<wave>"
DEFAULT_WAVE_PATCH_TOKEN = "<wave_patch>"
DEFAULT_WAVE_START_TOKEN = "<wave_bos>"
DEFAULT_WAVE_END_TOKEN = "<wave_eos>"
DEFAULT_WAVE_TOKEN_LEN = 248

# Default configuration for real dataset
DEFAULT_REAL_CONFIG = {
    'max_motion_length': 496,
    'min_motion_len': 96,
    'unit_length': 16,
    'raw': True,
    'thresholding': True,
}

# Default question prompts
DEFAULT_QUESTION_PROMPTS = [
    "Describe the human motion based on the provided wave signal representing joint movements.",
    "Interpret the wave signal and explain the corresponding human motion.",
    "Analyze the given wave signal to describe the overall movement of the human body.",
    "Given a wave signal representing motion data, explain how a human is moving.",
    "Use the provided wave signal to describe the sequence of human body movements.",
    "Interpret the wave signal to describe the motion pattern of the human body.",
    "Based on the wave signal data, explain the human motion being represented.",
    "Analyze the wave signal to generate a description of human motion.",
    "Use the wave signal to interpret and describe the human body's movement.",
    "Given the wave signal, describe the motion of the human body in detail."
]

@dataclass
class DataCollatorForWaveTextDataset(object):
    """Collate examples for mixed dataset with text and wave data."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "wave_token" in instances[0].keys():
            wave = [instance['wave_token'] for instance in instances]
            batch['input_wave_tokens'] = torch.stack(wave)
        
        if "wave_embed" in instances[0].keys():
            wave = [instance['wave_embed'] for instance in instances]
            batch['input_wave_embeds'] = torch.stack(wave)
            
        return batch


def preprocess_multimodal_wave(
    sources: Sequence[str],
    wave_indicator: str = DEFAULT_WAVE_INDICATOR,
    default_wave_patch_token: str = DEFAULT_WAVE_PATCH_TOKEN,
    wave_token_len: int = DEFAULT_WAVE_TOKEN_LEN,
    mm_use_wave_start_end: bool = True,
    default_wave_start_token: str = DEFAULT_WAVE_START_TOKEN,
    default_wave_end_token: str = DEFAULT_WAVE_END_TOKEN
) -> Dict:
    """Preprocess multimodal wave data by replacing wave indicators with tokens."""
    for source in sources:
        for sentence in source:
            replace_token = default_wave_patch_token * wave_token_len 
            if mm_use_wave_start_end:
                replace_token = default_wave_start_token + replace_token + default_wave_end_token
            if sentence["value"] is not None:
                sentence["value"] = sentence["value"].replace(wave_indicator, replace_token)

    return sources

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT, conv.sep_style

    # Mask targets
    sep = conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len + 1
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                print(' ==================== ')
                print(targets)
                print(input_ids)
                print(' ==================== ')
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch precess_v3: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def make_object_wave_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                  data_args):
    return {
        'train_dataset': WaveCaptionDataset(data_args.data_root,
                              split=data_args.split,
                              tokenizer=tokenizer),
        'eval_dataset':
        None,  # No validation dataset when split_train_val is False
        'data_collator': DataCollatorForWaveTextDataset(tokenizer)
    }

class WaveCaptionDataset(Dataset):
    """Dataset for wave caption tasks with multimodal support."""
    
    def __init__(self, data_root=None, split="train", tokenizer=None) -> None:
        super(WaveCaptionDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.tokenizer = tokenizer
        self.opt = self._load_config(data_root, split)
        self.data = self._load_data()
        
        print(f"load {len(self.data)} data as {self.split} set")
    
    def _load_config(self, data_root, split):
        """Load configuration based on split type."""
        if split == 'real':
            return edict(DEFAULT_REAL_CONFIG)
        else:
            config_path = osp.join(data_root, 'config.json')
            with open(config_path, 'r') as file:
                yaml_data = edict(json.load(file))
            return yaml_data.dataset_cfg
    
    def _get_filename(self, data_item, postfix):
        """Get filename for motion data with specific postfix."""
        file_path = os.path.join(data_item['filefolder'], data_item['fileindex'] + postfix)
        # debug: set the used file folder
        file_path = file_path.replace('udoppler', 'udoppler_rotation')
        if os.path.exists(file_path):
            return file_path
        raise FileNotFoundError(f"File {data_item['filefolder']}/{data_item['fileindex'] + postfix} not found")
    
    def _load_data(self):
        """Load and process dataset."""
        data_path = 'dataset/REAL/all.json' if self.split == 'real' else osp.join(self.data_root, f'{self.split}.json')
        
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
        """Create data items for all postfixes."""
        postfixes = ['A.npy', 'B.npy', 'C.npy', 'D.npy', 'E.npy', 'F.npy', 'G.npy', 'H.npy']
        data_items = []
        
        if 'test' in self.split:
            postfixes = [random.choice(postfixes)]
            
        for postfix in postfixes:
            data_items.append({
                'filename': self._get_filename(data_item, postfix),
                'question': qa.get('question') if qa else None,
                'answer': qa.get('answer') if qa else '',
                'caption': "\n".join(data_item['captions'])
            })
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
        instance = deepcopy(self.data[index])
        if instance['question'] is None:
            instance['question'] = self.default_question()
        sample = preprocess_multimodal_wave([self.format_caption(instance['question'], instance['answer'])])
        data_dict = preprocess(sample, self.tokenizer)
        motion = np.load(instance['filename'])
        motion, mask, m_length = process_motion(motion, self.opt)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0],
                        caption=instance['caption'],
                        question=instance['question'],
                        answer=instance['answer'],
                        wave_embed=torch.tensor(motion).unsqueeze(0).float())
        return data_dict
    

def mini_dataset(filename, tokenizer, question, answer, caption):
    """Create a mini dataset sample for testing purposes."""
    sample = preprocess_multimodal_wave([WaveCaptionDataset.format_caption(question, answer)])
    data_dict = preprocess(sample, tokenizer)
    motion = np.load(filename)
    opt = edict(DEFAULT_REAL_CONFIG)
    motion, mask, m_length = process_motion(motion, opt)
    
    return dict(
        input_ids=data_dict["input_ids"][0],
        labels=data_dict["labels"][0],
        caption=caption,
        question=question,
        answer=answer,
        wave_embed=torch.tensor(motion).unsqueeze(0).float()
    )


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
    from src.wavellm import conversation as conversation_lib
    from src.wavellm.conversation import conv_templates
    
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
    print("\n=== TEST MODE ===")
    test_dataset = WaveCaptionDataset(data_root=data_root, split="test_QAs", tokenizer=tokenizer)
    
    test_item = test_dataset[0]
    print("---")
    print(f"Question: {test_item['question']}")
    print(f"Answer: {test_item['answer']}")
    
    test_item = test_dataset[1]
    print("---")
    print(f"Question: {test_item['question']}")
    print(f"Answer: {test_item['answer']}")
    
    test_item = test_dataset[2]
    print("---")
    print(f"Question: {test_item['question']}")
    print(f"Answer: {test_item['answer']}")
    