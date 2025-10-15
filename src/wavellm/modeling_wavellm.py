import json
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Phi3Model, Phi3ForCausalLM
from typing import List, Optional, Tuple, Union
from contextlib import nullcontext

from src.misc.tools import instantiate_from_config


from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

class WaveLLM(Phi3Model):
    def __init__(self, config):
        super(WaveLLM, self).__init__(config)
        # Initialize weights and apply final processing

        self.vae_latent_dim = config.vae_latent_dim
        self.mm_projection_layers = nn.Linear(self.vae_latent_dim,self.config.hidden_size)
        self.wave_token = getattr(config,'wave_token','<|wave_token|>')
                
        self.post_init()

    def load_wave_encoder(self, encoder_version_path):
        encoder_cfg = {
            'target': 'src.model.clip.ImageEncoder',
            'params': json.load(open(f'{encoder_version_path}/config.json'))['encoder_cfg']}
        encoder = instantiate_from_config(encoder_cfg).eval()
        weights = torch.load(f'{encoder_version_path}/encoder.pth')
        encoder.load_state_dict(weights, strict=True)
        for p in encoder.parameters():
            p.requires_grad = False
        self.wave_encoder = encoder

    def forward(
        self,
        input_wave_tokens: torch.LongTensor = None,
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
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if input_ids.shape[1] != 1 or self.training:
            if input_wave_tokens is not None or input_wave_embeds is not None:
                bs = input_ids.shape[0]
                wave_features = self.mm_projection_layers(self.wave_encoder.encode_to_sequence(input_wave_embeds))

                new_input_embeds = []
                cur_wave_idx = 0
                for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C
                    
                    cur_wave_features = wave_features[cur_wave_idx].to(device=cur_input_embeds.device)
                    num_patches = cur_wave_features.shape[0] # * number of wave tokens

                    if self.config.mm_use_wave_start_end:
                        if (cur_input_ids == self.config.wave_start_token).sum() != (cur_input_ids == self.config.wave_end_token).sum():
                            raise ValueError("The number of wave start tokens and wave end tokens should be the same.")
                        wave_start_tokens = torch.where(cur_input_ids == self.config.wave_start_token)[0]
                        for wave_start_token_pos in wave_start_tokens:
                            if cur_input_ids[wave_start_token_pos + num_patches + 1] != self.config.wave_end_token:
                                raise ValueError("The wave end token should follow the wave start token.")
                            if orig_embeds_params is not None: # * will not update the original embeddings except for wave_START_TOKEN and wave_END_TOKEN
                                cur_new_input_embeds = torch.cat((cur_input_embeds[:wave_start_token_pos].detach(), cur_input_embeds[wave_start_token_pos:wave_start_token_pos+1], cur_wave_features, cur_input_embeds[wave_start_token_pos + num_patches + 1:wave_start_token_pos + num_patches + 2], cur_input_embeds[wave_start_token_pos + num_patches + 2:].detach()), dim=0)
                            else:
                                cur_new_input_embeds = torch.cat((cur_input_embeds[:wave_start_token_pos+1], cur_wave_features, cur_input_embeds[wave_start_token_pos + num_patches + 1:]), dim=0)
                            cur_wave_idx += 1
                            
                        new_input_embeds.append(cur_new_input_embeds)
                    else:
                        if (cur_input_ids == self.config.wave_patch_token).sum() != num_patches:
                            raise ValueError("The number of wave patch tokens should be the same as the number of wave patches.")
                        masked_indices = torch.where(cur_input_ids == self.config.wave_patch_token)[0]
                        mask_index_start = masked_indices[0]
                        if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                            raise ValueError("The wave patch tokens should be consecutive.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_wave_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_wave_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                        new_input_embeds.append(cur_new_input_embeds)
                        cur_wave_idx += 1
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
            else: 
                raise ValueError("Either input_wave_tokens or input_wave_embeds should be provided.")

        return super(WaveLLM, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

class WaveLLMForCausalLM(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = WaveLLM(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=0,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids = input_ids, 
            past_key_values = past_key_values, 
            attention_mask = attention_mask, 
            inputs_embeds=inputs_embeds,
            cache_position=cache_position, 
            position_ids=position_ids, 
            use_cache=use_cache, 
            num_logits_to_keep=num_logits_to_keep,
            **kwargs)
        model_inputs.update({
            "input_wave_tokens": kwargs.get("input_wave_tokens", None),
            "input_wave_embeds": kwargs.get("input_wave_embeds", None),
        })
        return model_inputs
    
    
    def forward(
        self,
        input_wave_tokens: torch.LongTensor = None,
        input_wave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None, # * control whether to return past_key_values
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        position_ids = None

        outputs = self.model(input_ids=input_ids,
                             input_wave_tokens=input_wave_tokens,
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
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # * B, L, V(32003)
            shift_labels = labels[..., 1:].contiguous() # * B, L
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
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
    
    def initialize_tokenizer_wave_backbone_config(self, tokenizer, device, fix_llm=True):

        config = self.config
        mm_use_wave_start_end = self.config.mm_use_wave_start_end = config.mm_use_wave_start_end

        default_wave_patch_token = config.default_wave_patch_token
        self.config.default_wave_patch_token = default_wave_patch_token
        tokenizer.add_tokens([default_wave_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        self.config.wave_patch_token = tokenizer.convert_tokens_to_ids([default_wave_patch_token])[0]

        if mm_use_wave_start_end:
            default_wave_start_token = config.default_wave_start_token
            default_wave_end_token = config.default_wave_end_token
            self.config.default_wave_start_token = default_wave_start_token
            self.config.default_wave_end_token = default_wave_end_token

            num_new_tokens = tokenizer.add_tokens([default_wave_start_token, default_wave_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            self.config.wave_start_token = tokenizer.convert_tokens_to_ids([default_wave_start_token])[0]
            self.config.wave_end_token = tokenizer.convert_tokens_to_ids([default_wave_end_token])[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")