"""Base model classes for models with mmwave feature support"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import PreTrainedModel
from typing import List, Optional, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from termcolor import colored


class WaveModelBase(PreTrainedModel, ABC):
    """Abstract base class for models with wave feature support"""

    def process_wave_features(
        self,
        inputs_embeds: torch.FloatTensor,
        input_wave_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Replace wave_patch_tokens with mmwave feature embeddings

        Args:
            inputs_embeds: Text embeddings [B, L, H]
            input_wave_embeds: Mmwave features [B, T, H] (already projected)
            input_ids: Input token ids [B, L]
            attention_mask: Original attention mask [B, L] or None
        Returns:
            Combined embeddings [B, L_new, H] and updated attention mask [B, L_new]
        """
        batch_size = input_ids.shape[0]
        wave_patch_token_id = getattr(self.config, 'wave_patch_token', None)
        wave_token_len = getattr(self.config, 'wave_token_len', 248)

        # Get device from input tensors
        device = inputs_embeds.device

        # Create default attention_mask if None
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

        # Process each sample in the batch
        new_embeds = []
        new_masks = []

        for batch_idx in range(batch_size):
            sample_embeds = []
            sample_mask = []
            wave_idx = 0

            for seq_idx in range(input_ids.shape[1]):
                current_token_id = input_ids[batch_idx, seq_idx].item()

                if current_token_id == wave_patch_token_id:
                    # Replace this wave_patch_token with mmwave embeddings
                    if wave_idx < input_wave_embeds.shape[1]:
                        sample_embeds.append(input_wave_embeds[batch_idx, wave_idx])
                        sample_mask.append(1)
                        wave_idx += 1
                    else:
                        # No more wave features available, keep original
                        sample_embeds.append(inputs_embeds[batch_idx, seq_idx])
                        sample_mask.append(attention_mask[batch_idx, seq_idx].item())
                else:
                    # Regular token, keep original embedding
                    sample_embeds.append(inputs_embeds[batch_idx, seq_idx])
                    sample_mask.append(attention_mask[batch_idx, seq_idx].item())

            new_embeds.append(torch.stack(sample_embeds))
            new_masks.append(torch.tensor(sample_mask, dtype=torch.long, device=device))

        result_embeds = torch.stack(new_embeds)
        result_masks = torch.stack(new_masks)

        return result_embeds, result_masks

    @abstractmethod
    def _get_base_model_class(self):
        """Get the base model class (Phi3Model or Qwen2Model)"""
        pass

    def forward(
        self,
        input_wave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """Forward with wave feature fusion, matching old version's logic"""
        # Embed input_ids if inputs_embeds not provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Fuse wave features on first forward (input_ids.shape[1] != 1) or during training
        # This matches the old version's condition exactly
        fusion_condition = input_ids.shape[1] != 1 or self.training

        if fusion_condition:
            if input_wave_embeds is not None:
                # Fuse wave features
                inputs_embeds, _ = self.process_wave_features(
                    inputs_embeds=inputs_embeds,
                    input_wave_embeds=input_wave_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            else:
                raise ValueError("input_wave_embeds is required for fusion")

        # Get the base class to call its forward method
        base_class = self._get_base_model_class()
        return super(base_class, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )


class WaveModelForCausalBase(WaveModelBase, ABC):
    """Abstract base class for causal language models with wave feature support"""

    def __init__(self, config, **kwargs):
        # Extract dtype from kwargs for compatibility with newer transformers
        dtype = kwargs.pop('dtype', None)

        # Initialize the parent CausalLM model (this will be called by concrete subclasses)
        super().__init__(config, **kwargs)

        # Convert model to specified dtype if provided (for compatibility)
        if dtype is not None:
            self.to(dtype=dtype)

        # mm_projection_layers will be initialized later via initialize_wave_projection()
        self.mm_projection_layers = None

    @abstractmethod
    def _get_base_model_class(self):
        """Get the base model class (Phi3ForCausalLM or Qwen2ForCausalLM)"""
        pass

    def initialize_wave_projection(self, wave_feature_dim: int):
        """Initialize wave feature projection layer

        Args:
            wave_feature_dim: Dimension of wave features (radar encoder output)
        """
        if self.mm_projection_layers is not None:
            # Already initialized
            return

        # Create projection layer with model's dtype
        model_dtype = next(self.parameters()).dtype
        self.mm_projection_layers = nn.Linear(
            wave_feature_dim,
            self.config.hidden_size
        ).to(dtype=model_dtype)

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """Prepare inputs for generation, pass through input_wave_embeds"""
        input_wave_embeds = kwargs.get("input_wave_embeds", None)

        # Get the base class to call its prepare_inputs_for_generation method
        base_class = self._get_base_model_class()
        model_inputs = super(base_class, self).prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )

        # Pass through input_wave_embeds to forward
        model_inputs.update({
            "input_wave_embeds": input_wave_embeds,
        })

        return model_inputs

    def forward(
        self,
        input_wave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward matching old version's WaveLLMForCausalLM"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        position_ids = None

        # Call self.model (custom model) which handles wave fusion
        outputs = self.model(
            input_ids=input_ids,
            input_wave_embeds=input_wave_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )