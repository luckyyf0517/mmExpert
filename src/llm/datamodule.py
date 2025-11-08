"""DataModule for mmwave feature + Q&A training with conversation templates"""

import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from easydict import EasyDict
from src.logger import log_message

# Constants for wave tokens
DEFAULT_WAVE_INDICATOR = "<wave>"
DEFAULT_WAVE_PATCH_TOKEN = "<wave_patch>"
DEFAULT_WAVE_START_TOKEN = "<wave_bos>"
DEFAULT_WAVE_END_TOKEN = "<wave_eos>"
DEFAULT_WAVE_TOKEN_LEN = 248
IGNORE_INDEX = -100

# Import conversation utilities
from src.llm.utils import conversation as conversation_lib


def preprocess_multimodal_wave(
    sources,
    wave_indicator=DEFAULT_WAVE_INDICATOR,
    default_wave_patch_token=DEFAULT_WAVE_PATCH_TOKEN,
    wave_token_len=DEFAULT_WAVE_TOKEN_LEN,
    mm_use_wave_start_end=True,
    default_wave_start_token=DEFAULT_WAVE_START_TOKEN,
    default_wave_end_token=DEFAULT_WAVE_END_TOKEN
):
    """Preprocess multimodal wave data by replacing wave indicators with tokens."""
    for source in sources:
        for sentence in source:
            replace_token = default_wave_patch_token * wave_token_len
            if mm_use_wave_start_end:
                replace_token = default_wave_start_token + replace_token + default_wave_end_token
            if sentence["value"] is not None:
                sentence["value"] = sentence["value"].replace(wave_indicator, replace_token)
    return sources


def preprocess_with_conversation(sources, tokenizer, conversation_template="conv_phi3"):
    """Process conversation data using conversation templates."""
    # Set conversation template
    conversation_lib.default_conversation = conversation_lib.conv_templates[conversation_template].copy()
    conv = conversation_lib.default_conversation

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
            assert role == conv.roles[j % 2], f"Conversation format error at {i}"
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
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT, f"Expected MPT style, got {conv.sep_style}"

    # Mask targets for loss computation
    sep = conv.roles[1]  # assistant role
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))  # user + gpt

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

            # Mask instruction part (only compute loss on assistant response)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len + 1

        target[cur_len:] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "labels": targets,
    }


class WaveLLMDataModule(pl.LightningDataModule):
    """DataModule for mmwave feature + Q&A training with conversation templates"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Data configuration
        self.data_root = cfg.data_root
        self.model_path = cfg.model_path
        self.conversation_template = cfg.get('conversation_template', 'conv_phi3')
        self.model_max_length = cfg.get('model_max_length', 2048)
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.get('num_workers', 4)
        self.train_split = cfg.train_split
        self.test_split = cfg.test_split

        # Tokenizer will be set in setup
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        # Wave token configuration
        self.mm_use_wave_start_end = cfg.get('mm_use_wave_start_end', True)
        self.wave_token_len = cfg.get('wave_token_len', 248)

    def setup(self, stage=None):
        """Setup datasets"""
        # Initialize tokenizer with conversation template
        self.tokenizer = self._load_tokenizer()
        self._setup_wave_tokens()

        # Set conversation template globally
        conversation_lib.default_conversation = conversation_lib.conv_templates[self.conversation_template].copy()

        # Load datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = self._load_dataset(self.train_split)
            self.eval_dataset = self._load_dataset(self.test_split) if stage == 'fit' else None

        if stage == 'test':
            self.test_dataset = self._load_dataset(self.test_split)

    def _load_tokenizer(self):
        """Load tokenizer from model path"""
        # Convert to absolute path
        model_path = os.path.abspath(self.model_path)
        
        # Check if path exists and is a directory
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path is not a directory: {model_path}")
        
        # Load tokenizer from local files only (no fallback to hub)
        load_kwargs = {
            'model_max_length': self.model_max_length,
            'padding_side': "right",
            'use_fast': False,
            'local_files_only': True
        }
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _setup_wave_tokens(self):
        """Setup wave tokens in tokenizer"""
        # Add wave patch token
        num_new_tokens = self.tokenizer.add_tokens([DEFAULT_WAVE_PATCH_TOKEN], special_tokens=True)

        # Add start/end tokens if needed
        if self.mm_use_wave_start_end:
            start_end_tokens = [DEFAULT_WAVE_START_TOKEN, DEFAULT_WAVE_END_TOKEN]
            num_new_tokens += self.tokenizer.add_tokens(start_end_tokens, special_tokens=True)

        log_message("INFO", f"Added {num_new_tokens} wave tokens to tokenizer", color="green")

    def _format_conversation(self, question, answer):
        """Format question and answer into conversation format with wave indicator."""
        return [{
            "from": "human",
            "value": f"{DEFAULT_WAVE_INDICATOR}\n{question}"
        }, {
            "from": "gpt",
            "value": answer
        }]

    def _load_dataset(self, split):
        """Load dataset using existing wavellm_dataset logic but with new conversation handling
        
        Returns a wrapper dataset that processes items on-the-fly to avoid loading all data into memory.
        """
        # Import existing dataset class
        from .dataset import WaveCaptionDataset
        from torch.utils.data import Dataset as TorchDataset
        
        # Create original dataset to get raw data (this just loads metadata, not NPZ files)
        original_dataset = WaveCaptionDataset(
            data_root=self.data_root,
            split=split,
            tokenizer=self.tokenizer
        )
        
        # Create a wrapper dataset that processes items on-the-fly
        class ConversationDatasetWrapper(TorchDataset):
            def __init__(self, base_dataset, format_conversation_fn, 
                        preprocess_multimodal_wave_fn, preprocess_with_conversation_fn,
                        tokenizer, conversation_template, mm_use_wave_start_end, wave_token_len):
                self.base_dataset = base_dataset
                self.format_conversation = format_conversation_fn
                self.preprocess_multimodal_wave = preprocess_multimodal_wave_fn
                self.preprocess_with_conversation = preprocess_with_conversation_fn
                self.tokenizer = tokenizer
                self.conversation_template = conversation_template
                self.mm_use_wave_start_end = mm_use_wave_start_end
                self.wave_token_len = wave_token_len
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, index):
                # Get item from base dataset (this loads NPZ file on-demand)
                item = self.base_dataset[index]
                
                # Use the existing question/answer format
                question = item['question']
                answer = item['answer']
                filename = item['filename']
                
                # Format as conversation
                conversation = self.format_conversation(question, answer)
                
                # Preprocess with wave tokens
                processed_conversation = self.preprocess_multimodal_wave(
                    [conversation],
                    mm_use_wave_start_end=self.mm_use_wave_start_end,
                    wave_token_len=self.wave_token_len
                )
                
                # Process with conversation template
                processed_data = self.preprocess_with_conversation(
                    processed_conversation,
                    self.tokenizer,
                    self.conversation_template
                )
                
                return {
                    'input_ids': processed_data["input_ids"][0],
                    'labels': processed_data["labels"][0],
                    'wave_embed': item['wave_embed'],  # Keep the original wave embedding
                    'question': question,
                    'answer': answer,
                    'filename': filename
                }
        
        # Return wrapper dataset (lazy loading)
        return ConversationDatasetWrapper(
            original_dataset,
            self._format_conversation,
            preprocess_multimodal_wave,
            preprocess_with_conversation,
            self.tokenizer,
            self.conversation_template,
            self.mm_use_wave_start_end,
            self.wave_token_len
        )

    def _collate_fn(self, batch):
        """Collate function for batches"""
        # Extract features
        wave_embeds = [item['wave_embed'] for item in batch]
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Extract question and answer if available (for evaluation)
        questions = [item.get('question', '') for item in batch]
        answers = [item.get('answer', '') for item in batch]
        filenames = [item.get('filename', '') for item in batch]

        # Stack wave_embeds into [B, H, W] format for each view
        # Find max time dimension (W) for each view to pad to same length
        batch_size = len(wave_embeds)
        
        # Get max time dimension for each view
        max_range_time = max(embed['range_time'].shape[1] for embed in wave_embeds)
        max_doppler_time = max(embed['doppler_time'].shape[1] for embed in wave_embeds)
        max_azimuth_time = max(embed['azimuth_time'].shape[1] for embed in wave_embeds)
        
        # Stack and pad each view to [B, H, W]
        stacked_range_time = []
        stacked_doppler_time = []
        stacked_azimuth_time = []
        
        for embed in wave_embeds:
            # Pad range_time to max time dimension
            range_time = embed['range_time']  # [H, W]
            if range_time.shape[1] < max_range_time:
                pad_w = max_range_time - range_time.shape[1]
                range_time = torch.cat([range_time, torch.zeros(range_time.shape[0], pad_w, dtype=range_time.dtype, device=range_time.device)], dim=1)
            stacked_range_time.append(range_time)
            
            # Pad doppler_time
            doppler_time = embed['doppler_time']  # [H, W]
            if doppler_time.shape[1] < max_doppler_time:
                pad_w = max_doppler_time - doppler_time.shape[1]
                doppler_time = torch.cat([doppler_time, torch.zeros(doppler_time.shape[0], pad_w, dtype=doppler_time.dtype, device=doppler_time.device)], dim=1)
            stacked_doppler_time.append(doppler_time)
            
            # Pad azimuth_time
            azimuth_time = embed['azimuth_time']  # [H, W]
            if azimuth_time.shape[1] < max_azimuth_time:
                pad_w = max_azimuth_time - azimuth_time.shape[1]
                azimuth_time = torch.cat([azimuth_time, torch.zeros(azimuth_time.shape[0], pad_w, dtype=azimuth_time.dtype, device=azimuth_time.device)], dim=1)
            stacked_azimuth_time.append(azimuth_time)
        
        # Stack to [B, H, W] format
        mmwave_features = {
            'range_time': torch.stack(stacked_range_time),  # [B, H, W]
            'doppler_time': torch.stack(stacked_doppler_time),  # [B, H, W]
            'azimuth_time': torch.stack(stacked_azimuth_time)  # [B, H, W]
        }

        # Pad sequences to max length in batch
        max_len = max(len(ids) for ids in input_ids)

        # Pad input_ids and labels
        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for i, (ids, lbl) in enumerate(zip(input_ids, labels)):
            current_len = len(ids)
            if current_len < max_len:
                # Pad with pad_token_id for input_ids and -100 for labels
                padding_length = max_len - current_len
                padded_ids = ids.tolist() + [self.tokenizer.pad_token_id] * padding_length
                padded_lbl = lbl.tolist() + [-100] * padding_length
                attention_mask = [1] * current_len + [0] * padding_length
            else:
                padded_ids = ids.tolist()
                padded_lbl = lbl.tolist()
                attention_mask = [1] * max_len

            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbl)
            attention_masks.append(attention_mask)

        result = {
            'mmwave_features': mmwave_features,  # Dict with [B, H, W] tensors
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }
        
        # Add question and answer if available (for evaluation)
        result['questions'] = questions
        result['answers'] = answers
        result['filenames'] = filenames
        
        return result

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        if self.eval_dataset is None:
            return None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )


