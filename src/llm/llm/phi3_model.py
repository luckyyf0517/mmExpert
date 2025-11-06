import torch
import torch.nn as nn
from transformers import Phi3ForCausalLM
from typing import List, Optional, Union, Dict, Any, Tuple
from .base_model import WaveBaseModel

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

class Phi3ForCausalLM(WaveBaseModel, Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def get_output_embeddings(self):
        # Return lm_head for output embeddings
        return getattr(self, "lm_head", None)

    def forward(
        self,
        input_wave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # Process input embeddings
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        if input_wave_embeds is not None:
            inputs_embeds, attention_mask = self.process_wave_features(inputs_embeds, input_wave_embeds)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_wave_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ):
        assert input_wave_embeds is not None and input_ids is not None, \
            "input_wave_embeds and input_ids must be provided"
            
        inputs_embeds = self.model.embed_tokens(input_ids)
        inputs_embeds, attention_mask = self.process_wave_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_wave_embeds=input_wave_embeds,
            attention_mask=attention_mask
        )
        
        return super().generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True, 
            **model_kwargs
        )
