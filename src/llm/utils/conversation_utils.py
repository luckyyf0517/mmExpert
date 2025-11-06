"""Utilities for handling conversation templates with mmwave features"""

import sys
import os.path as osp
sys.path.append('.')

from . import conversation as conversation_lib
from .conversation import conv_templates, SeparatorStyle


class ConversationHandler:
    """Handler for conversation template processing with mmwave features"""

    def __init__(self, conv_template="conv_phi3", wave_indicator="<wave>"):
        """Initialize conversation handler

        Args:
            conv_template: Name of conversation template
            wave_indicator: Wave indicator token to replace
        """
        self.conv_template = conv_template
        self.wave_indicator = wave_indicator

        # Set default conversation
        conversation_lib.default_conversation = conv_templates[conv_template]
        self.conv = conv_templates[conv_template].copy()

    def format_qa_conversation(self, question, answer, use_wave_indicator=True):
        """Format Q&A into conversation template

        Args:
            question: Question text
            answer: Answer text
            use_wave_indicator: Whether to include wave indicator

        Returns:
            Formatted conversation text
        """
        # Reset conversation
        self.conv.messages = []

        # Add human message
        if use_wave_indicator:
            human_msg = f"{self.wave_indicator}\n{question}"
        else:
            human_msg = question

        self.conv.append_message(self.conv.roles[0], human_msg)
        self.conv.append_message(self.conv.roles[1], answer)

        return self.conv.get_prompt()

    def format_qa_with_wave_patch(self, question, answer):
        """Format Q&A with wave_patch_token for training

        Args:
            question: Question text
            answer: Answer text

        Returns:
            Formatted conversation with wave_patch_token
        """
        return self.format_qa_conversation(question, answer, use_wave_indicator=True)

    def format_qa_without_wave(self, question, answer):
        """Format Q&A without wave indicator for inference only

        Args:
            question: Question text
            answer: Answer text

        Returns:
            Formatted conversation without wave indicator
        """
        return self.format_qa_conversation(question, answer, use_wave_indicator=False)

    def replace_wave_indicator_with_tokens(self, text, wave_patch_token="<wave_patch>", wave_token_len=248):
        """Replace wave indicators with actual wave tokens

        Args:
            text: Input text containing <wave> indicators
            wave_patch_token: Token to replace <wave> with
            wave_token_len: Number of wave_patch_tokens to use

        Returns:
            Text with wave indicators replaced by wave_patch_tokens
        """
        # Check if start/end tokens should be used
        use_start_end = hasattr(self.conv, 'mm_use_wave_start_end') and self.conv.mm_use_wave_start_end

        if use_start_end:
            wave_bos_token = getattr(self.conv, 'wave_bos_token', "<wave_bos>")
            wave_eos_token = getattr(self.conv, 'wave_eos_token', "<wave_eos>")
            replacement = f"{wave_bos_token} {wave_patch_token * wave_token_len} {wave_eos_token}"
        else:
            replacement = f"{wave_patch_token * wave_token_len}"

        return text.replace(self.wave_indicator, replacement)

    def preprocess_for_training(self, sources, tokenizer):
        """Preprocess conversation sources for training

        Args:
            sources: List of conversation sources
            tokenizer: Tokenizer instance

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        # Apply conversation template
        conversations = []
        for source in sources:
            conv = self.conv.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"Invalid conversation format"
                conv.append_message(role, sentence["value"])

            # Replace wave indicators with tokens
            prompt = self.replace_wave_indicator_with_tokens(conv.get_prompt())
            conversations.append(prompt)

        # Tokenize conversations
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        # Create labels
        targets = input_ids.clone()

        # Mask targets according to conversation template
        if self.conv.sep_style == SeparatorStyle.MPT:
            return self._mask_targets_mpt(conversations, targets, tokenizer)
        else:
            raise ValueError(f"Unsupported separator style: {self.conv.sep_style}")

    def _mask_targets_mpt(self, conversations, targets, tokenizer):
        """Mask targets for MPT style conversation"""
        assert self.conv.sep_style == SeparatorStyle.MPT
        sep = self.conv.roles[1]
        ignore_index = -100

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            rounds = conversation.split(self.conv.sep)
            re_rounds = [self.conv.sep.join(rounds[:3])]  # system + user + gpt

            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(self.conv.sep.join(rounds[conv_idx:conv_idx+2]))  # user + gpt

            cur_len = 1
            target[:cur_len] = ignore_index
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    continue

                parts = rou.split(sep)
                if len(parts) != 2:
                    break

                parts[0] += sep
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)

                target[cur_len: cur_len + instruction_len] = ignore_index
                cur_len += round_len

            target[cur_len:] = ignore_index

        return dict(
            input_ids=targets,
            labels=targets.clone(),
            attention_mask=targets.ne(tokenizer.pad_token_id)
        )