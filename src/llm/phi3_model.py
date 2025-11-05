"""Phi3 model with mmwave feature support"""

import torch
from transformers import Phi3ForCausalLM as HFForPhi3ForCausalLM
from .base_model import WaveBaseModel


class Phi3ForCausalLM(WaveBaseModel, HFForPhi3ForCausalLM):
    """Phi3 model with mmwave feature support"""

    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def forward(
        self,
        mmwave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        **kwargs
    ):
        # Prepare text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Combine mmwave and text features using conversation template
        if mmwave_embeds is not None and input_ids is not None:
            inputs_embeds, attention_mask = self.process_wave_features(
                inputs_embeds, mmwave_embeds, input_ids, attention_mask
            )

            # Adjust labels if provided
            if labels is not None:
                labels = self._adjust_labels_for_wave_tokens(labels, input_ids)

        # Forward pass
        return super().forward(
            input_ids=None,  # We're using inputs_embeds directly
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def _adjust_labels_for_wave_tokens(self, labels, input_ids):
        """Adjust labels by replacing wave_patch_token positions with -100"""
        wave_patch_token_id = getattr(self.config, 'wave_patch_token', None)

        if wave_patch_token_id is None:
            return labels

        # Create a mask for wave patch tokens
        wave_mask = (input_ids == wave_patch_token_id)

        # Create adjusted labels where wave_patch_tokens are replaced with -100
        adjusted_labels = labels.clone()
        adjusted_labels[wave_mask] = -100

        return adjusted_labels