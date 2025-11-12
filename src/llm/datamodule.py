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


def preprocess_with_conversation(sources, tokenizer, conversation_template="conv_phi3", is_inference=False):
    """Process conversation data using conversation templates.
    
    Args:
        sources: List of conversation sources
        tokenizer: Tokenizer instance
        conversation_template: Template name (default: "conv_phi3")
        is_inference: If True, append assistant role token to prompt for generation
    """
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
        
        prompt = conv.get_prompt()
        
        # In inference mode, append assistant role token to guide generation
        # This ensures model knows to generate assistant response, not user message
        if is_inference:
            prompt += conv.roles[1]  # Add "<|assistant|>\n"
        
        conversations.append(prompt)

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
        
        # Wave feature embedding configuration
        self.embed_wave_features = cfg.get('embed_wave_features', True)
        """Whether to embed wave features into the model.
        If True: Wave features are processed and returned in batch (CausalLM approach)
        If False: Wave features are not embedded, can be used for Vision2Seq wrapper approach
        """

        # Inference mode flag
        self.is_inference = cfg.get('is_inference', False)

    def setup(self, stage='fit'):
        """Setup datasets"""
        # Initialize tokenizer with conversation template
        self.tokenizer = self._load_tokenizer()
        self._setup_wave_tokens()

        # Set conversation template globally
        conversation_lib.default_conversation = conversation_lib.conv_templates[self.conversation_template].copy()

        # Load datasets
        if stage == 'fit':
            self.train_dataset = self._load_dataset(self.train_split, stage='train')
            self.eval_dataset = self._load_dataset(self.test_split, stage='val')

        if stage == 'test':
            self.test_dataset = self._load_dataset(self.test_split, stage='test')

    def _load_tokenizer(self):
        """Load tokenizer from model path"""
        # Use model path directly (can be HuggingFace Hub identifier or local path)
        model_path = self.model_path
        
        # # Check if path exists and is a directory
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # if not os.path.isdir(model_path):
        #     raise ValueError(f"Model path is not a directory: {model_path}")
        
        # Load tokenizer from local files or hub (allow fallback to hub for model identifiers)
        load_kwargs = {
            'model_max_length': self.model_max_length,
            'padding_side': "right",
            'use_fast': False,
            'local_files_only': False
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
        conversation = [{
            "from": "human",
            "value": f"{DEFAULT_WAVE_INDICATOR}\n{question}"
        }]

        # Only add answer if not in inference mode
        if not self.is_inference:
            conversation.append({
                "from": "gpt",
                "value": answer
            })

        return conversation

    def _load_dataset(self, split, stage='train'):
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
            stage=stage,
            tokenizer=self.tokenizer,
            caption_only=self.cfg.get('caption_only', None),
            use_random_question_for_caption=self.cfg.get('use_random_question_for_caption', True)
        )
        
        # Create a wrapper dataset that processes items on-the-fly
        class ConversationDatasetWrapper(TorchDataset):
            def __init__(self, base_dataset, format_conversation_fn,
                        preprocess_multimodal_wave_fn, preprocess_with_conversation_fn,
                        tokenizer, conversation_template, mm_use_wave_start_end, 
                        wave_token_len, is_inference, embed_wave_features):
                self.base_dataset = base_dataset
                self.format_conversation = format_conversation_fn
                self.preprocess_multimodal_wave = preprocess_multimodal_wave_fn
                self.preprocess_with_conversation = preprocess_with_conversation_fn
                self.tokenizer = tokenizer
                self.conversation_template = conversation_template
                self.mm_use_wave_start_end = mm_use_wave_start_end
                self.wave_token_len = wave_token_len
                self.is_inference = is_inference
                self.embed_wave_features = embed_wave_features
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, index):
                # Get item from base dataset (this loads NPZ file on-demand)
                item = self.base_dataset[index]

                # Use the existing question/answer format
                question = item['question']
                answer = item['answer']
                filename = item['filename']

                # Format as conversation (format_conversation handles inference mode by not adding answer to conversation)
                conversation = self.format_conversation(question, answer)

                # Preprocess with wave tokens (same for both training and inference)
                # Only add wave tokens if embed_wave_features is True
                if self.embed_wave_features:
                    processed_conversation = self.preprocess_multimodal_wave(
                        [conversation],
                        mm_use_wave_start_end=self.mm_use_wave_start_end,
                        wave_token_len=self.wave_token_len
                    )
                else:
                    # Remove wave indicator without adding tokens (for Vision2Seq wrapper approach)
                    processed_conversation = [conversation]
                    for source in processed_conversation:
                        for sentence in source:
                            if sentence["value"] is not None:
                                sentence["value"] = sentence["value"].replace(DEFAULT_WAVE_INDICATOR, "")

                # Process with conversation template
                processed_data = self.preprocess_with_conversation(
                    processed_conversation,
                    self.tokenizer,
                    self.conversation_template,
                    is_inference=self.is_inference
                )

                input_ids = processed_data["input_ids"][0]
                labels = processed_data["labels"][0]

                return {
                    'input_ids': input_ids,
                    'labels': labels,
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
            self.wave_token_len,
            self.is_inference,
            self.embed_wave_features
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

    def _pad_and_stack_wave_views(self, wave_embeds, view_names=['range_time', 'doppler_time', 'azimuth_time']):
        """
        Pad and stack wave embeddings for multiple views
        
        Args:
            wave_embeds: List of wave embedding dicts, each containing view tensors
            view_names: List of view names to process
            
        Returns:
            Dict with stacked and padded tensors for each view [B, H, W]
        """
        # Get max time dimension for each view
        max_dims = {}
        for view_name in view_names:
            max_dims[view_name] = max(embed[view_name].shape[1] for embed in wave_embeds)
        
        # Stack and pad each view
        stacked_views = {}
        for view_name in view_names:
            stacked_list = []
            max_time = max_dims[view_name]
            
            for embed in wave_embeds:
                view_tensor = embed[view_name]  # [H, W]
                if view_tensor.shape[1] < max_time:
                    pad_w = max_time - view_tensor.shape[1]
                    view_tensor = torch.cat([
                        view_tensor, 
                        torch.zeros(
                            view_tensor.shape[0], 
                            pad_w, 
                            dtype=view_tensor.dtype, 
                            device=view_tensor.device
                        )
                    ], dim=1)
                stacked_list.append(view_tensor)
            
            stacked_views[view_name] = torch.stack(stacked_list)  # [B, H, W]
        
        return stacked_views

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

        # Process wave features - pad and stack all views
        # Both embed_wave_features=True and False need padding for batching
        mmwave_features = self._pad_and_stack_wave_views(wave_embeds)

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

                if self.is_inference:
                    # Left padding for inference/generation
                    padded_ids = [self.tokenizer.pad_token_id] * padding_length + ids.tolist()
                    padded_lbl = [-100] * padding_length + lbl.tolist()
                    attention_mask = [0] * padding_length + [1] * current_len
                else:
                    # Right padding for training
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


