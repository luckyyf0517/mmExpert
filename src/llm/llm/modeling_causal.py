"""Generic causal language model with wave feature fusion support

This module provides a generic implementation that can be used for
both Phi3 and Qwen models by parameterizing the base model classes.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple
from .utils import process_wave_features
from termcolor import colored

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)


def create_wave_causal_model(BaseModelClass):
    """Factory function to create a wave-enabled causal model class
    
    Args:
        BaseModelClass: The base model class from transformers (e.g., Phi3Model, Qwen2Model)
    
    Returns:
        A new class that extends BaseModelClass with wave feature fusion
    """
    class WaveCausalModel(BaseModelClass):
        """Custom causal model with wave feature fusion"""
        
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
            """Forward with wave feature fusion"""
            # Embed input_ids if inputs_embeds not provided
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            # Fuse wave features on first forward (input_ids.shape[1] != 1) or during training
            fusion_condition = input_ids.shape[1] != 1 or self.training

            if fusion_condition:
                if input_wave_embeds is not None:
                    # Fuse wave features
                    inputs_embeds, _ = process_wave_features(
                        inputs_embeds=inputs_embeds,
                        input_wave_embeds=input_wave_embeds,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        config=self.config
                    )
                else: 
                    raise ValueError("input_wave_embeds is required for fusion")
            
            # Call parent forward with inputs_embeds
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                position_ids=position_ids,
                **kwargs
            )
    
    return WaveCausalModel


def create_wave_causal_lm(BaseForCausalLMClass, BaseModelClass=None):
    """Factory function to create a wave-enabled causal LM class
    
    Args:
        BaseForCausalLMClass: The base ForCausalLM class from transformers 
                             (e.g., Phi3ForCausalLM, Qwen2ForCausalLM)
        BaseModelClass: Optional base model class. If not provided, will try to 
                       infer from BaseForCausalLMClass.base_model_class
    
    Returns:
        A new class that extends BaseForCausalLMClass with wave feature fusion
    """
    # Try to get base model class if not provided
    if BaseModelClass is None:
        if hasattr(BaseForCausalLMClass, 'base_model_class'):
            BaseModelClass = BaseForCausalLMClass.base_model_class
        else:
            # Try to infer from class name (e.g., Phi3ForCausalLM -> Phi3Model)
            # This is a fallback and may not work for all models
            raise ValueError(
                f"Cannot infer base model class from {BaseForCausalLMClass.__name__}. "
                "Please provide BaseModelClass explicitly."
            )
    
    # Create the wave-enabled model class
    WaveModelClass = create_wave_causal_model(BaseModelClass)
    
    class WaveCausalLM(BaseForCausalLMClass):
        """Custom causal LM with wave feature fusion"""
        
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            # Replace self.model with our wave-enabled model
            self.model = WaveModelClass(config)
            self.post_init()
            
            # mm_projection_layers will be initialized later via initialize_wave_projection()
            self.mm_projection_layers = None

        def initialize_wave_projection(self, wave_feature_dim: int):
            """Initialize wave feature projection layer
            
            Args:
                wave_feature_dim: Dimension of wave features (radar encoder output)
            """
            self.mm_projection_layers = nn.Linear(
                wave_feature_dim, 
                self.config.hidden_size
            )

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

            model_inputs = super().prepare_inputs_for_generation(
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
            """Forward with wave feature fusion"""
            
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            position_ids = None
            
            # Call self.model (wave-enabled model) which handles wave fusion
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
    
    return WaveCausalLM

