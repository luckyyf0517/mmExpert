import torch
import torch.nn as nn
from transformers import Phi3Model as _Phi3Model, Phi3ForCausalLM as _Phi3ForCausalLM
from typing import List, Optional, Union, Tuple
from .base_model import WaveBaseModel

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)


class Phi3Model(WaveBaseModel, _Phi3Model):
    """Custom Phi3Model that handles wave feature fusion"""
    
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
        if input_ids.shape[1] != 1 or self.training:
            if input_wave_embeds is not None:
                # Fuse wave features
                inputs_embeds, _ = self.process_wave_features(
                    inputs_embeds=inputs_embeds,
                    input_wave_embeds=input_wave_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        
        # Call parent Phi3Model.forward with input_ids=None (like old version)
        return super(Phi3Model, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )


class Phi3ForCausalLM(WaveBaseModel, _Phi3ForCausalLM):
    """Custom Phi3ForCausalLM that uses custom Phi3Model"""
    
    def __init__(self, config):
        super().__init__(config)
        # Replace self.model with our custom Phi3Model
        self.model = Phi3Model(config)
        self.post_init()
        
        # mm_projection_layers will be initialized later via initialize_wave_projection()
        self.mm_projection_layers = None

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
            "input_wave_embeds": kwargs.get("input_wave_embeds", None),
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
        
        # Call self.model (custom Phi3Model) which handles wave fusion
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
            **kwargs
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

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_wave_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ):
        """Generate with wave features"""
        assert input_wave_embeds is not None and input_ids is not None, \
            "input_wave_embeds and input_ids must be provided"
        
        # Create default attention_mask if None
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device
            )
        
        # Pass input_ids, input_wave_embeds to super().generate()
        # The model's forward will handle fusion
        return super().generate(
            input_ids=input_ids,
            input_wave_embeds=input_wave_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            **model_kwargs
        )
