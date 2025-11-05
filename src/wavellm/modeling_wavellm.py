import os
import json
import yaml
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Phi3Model, Phi3ForCausalLM
from typing import Dict, Any, List, Optional, Tuple, Union
from contextlib import nullcontext

from src.misc.tools import instantiate_from_config


from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

class WaveLLM(Phi3Model):
    """
    WaveLLM model with integrated radar encoder support.

    Extends Phi-3 with multimodal radar capabilities through dynamic encoder loading.
    Strict error handling - any configuration or runtime issues will raise exceptions.
    """

    def __init__(self, config):
        """Initialize WaveLLM model with radar encoder support."""
        super(WaveLLM, self).__init__(config)

        # Initialize dimension parameters
        self._initialize_dimensions(config)

        # Initialize projection layers (will be rebuilt after encoder loading)
        self._initialize_projection_layers()

        # Initialize wave token
        self._initialize_wave_token(config)

        # Initialize encoder tracking
        self._initialize_encoder_state()

        self.post_init()

    def _initialize_dimensions(self, config):
        """Initialize dimension parameters with strict validation."""
        self.mmwave_latent_dim = getattr(config, 'mmwave_latent_dim', getattr(config, 'vae_latent_dim', 512))

        if not isinstance(self.mmwave_latent_dim, int) or self.mmwave_latent_dim <= 0:
            raise ValueError(f"mmwave_latent_dim must be a positive integer, got {self.mmwave_latent_dim}")

        self.vae_latent_dim = self.mmwave_latent_dim  # Keep for backward compatibility

    def _initialize_projection_layers(self):
        """Initialize projection layers with current dimensions."""
        self.mm_projection_layers = nn.Linear(self.mmwave_latent_dim, self.config.hidden_size)

    def _initialize_wave_token(self, config):
        """Initialize wave token from configuration."""
        self.wave_token = getattr(config, 'wave_token', '<|wave_token|>')

    def _initialize_encoder_state(self):
        """Initialize encoder state tracking variables."""
        self._encoder_config = None
        self._actual_embed_dim = None

    def load_wave_encoder(self, encoder_version_path: str):
        """
        Load radar encoder from feature directory.

        Args:
            encoder_version_path: Path to feature directory containing encoder files

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If configuration is invalid
            RuntimeError: If encoder loading fails
        """
        # Validate input path
        if not isinstance(encoder_version_path, str):
            raise TypeError(f"encoder_version_path must be a string, got {type(encoder_version_path)}")

        if not os.path.exists(encoder_version_path):
            raise FileNotFoundError(f"Encoder directory not found: {encoder_version_path}")

        # Construct file paths
        config_path = os.path.join(encoder_version_path, 'radar_encoder_config.yaml')
        weights_path = os.path.join(encoder_version_path, 'radar_encoder.pth')

        # Validate required files exist
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Radar encoder config not found: {config_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Radar encoder weights not found: {weights_path}")

        # Load radar encoder
        return self._load_radar_encoder(encoder_version_path, config_path, weights_path)

    def _load_radar_encoder(self, encoder_version_path: str, config_path: str, weights_path: str):
        """
        Load radar encoder from YAML config and PyTorch weights.

        Strict error handling - any issues will raise exceptions immediately.

        Args:
            encoder_version_path: Path to encoder version directory
            config_path: Path to YAML configuration file
            weights_path: Path to PyTorch weights file

        Returns:
            Loaded and configured radar encoder
        """
        # Load and parse YAML configuration
        encoder_config = self._load_yaml_config(config_path)

        # Extract radar encoder configuration
        radar_cfg = self._extract_radar_config(encoder_config)

        # Store encoder config
        self._encoder_config = radar_cfg.copy()

        # Validate configuration
        self._validate_radar_config(radar_cfg)

        # Create encoder instance
        encoder = self._create_encoder_from_config(radar_cfg)

        # Load weights
        self._load_encoder_weights(encoder, weights_path)

        # Freeze parameters
        self._freeze_encoder_parameters(encoder)

        # Create wrapper for training pipeline
        wrapped_encoder = self._wrap_encoder_for_training(encoder)

        # Store encoder
        self.wave_encoder = wrapped_encoder.eval()

        # Update dimensions
        self._update_encoder_dimensions(encoder)

        # Log successful loading
        self._log_encoder_loading_success(encoder_version_path, radar_cfg)

        return encoder

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            from yaml import UnsafeLoader
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=UnsafeLoader)

        if config is None:
            raise ValueError(f"YAML config is empty: {config_path}")

        return config

    def _extract_radar_config(self, encoder_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract radar encoder configuration."""
        radar_cfg = encoder_config.get('radar_encoder_cfg')
        if radar_cfg is None:
            raise ValueError("radar_encoder_cfg not found in configuration")

        # Convert to dict if needed
        if hasattr(radar_cfg, 'to_dict'):
            radar_cfg = radar_cfg.to_dict()
        elif hasattr(radar_cfg, 'dict'):
            radar_cfg = radar_cfg.dict()
        elif not isinstance(radar_cfg, dict):
            raise ValueError(f"radar_cfg must be dict-like, got {type(radar_cfg)}")

        return radar_cfg

    def _validate_radar_config(self, radar_cfg: Dict[str, Any]) -> None:
        """Validate radar encoder configuration."""
        # Check required fields
        if 'embed_dim' not in radar_cfg:
            raise ValueError("embed_dim is required in radar config")

        embed_dim = radar_cfg['embed_dim']
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive integer, got {embed_dim}")

        # Validate optional fields
        if 'max_sequence_length' in radar_cfg:
            max_seq_len = radar_cfg['max_sequence_length']
            if not isinstance(max_seq_len, int) or max_seq_len <= 0:
                raise ValueError(f"max_sequence_length must be positive integer, got {max_seq_len}")

    def _create_encoder_from_config(self, radar_cfg: Dict[str, Any]):
        """Create encoder instance."""
        encoder_type = radar_cfg.get('encoder_type', 'radar_encoder_temporal')

        if encoder_type == 'radar_encoder_vit':
            from src.encoders.radar_encoder_vit import RadarEncoderViT
            return RadarEncoderViT(**radar_cfg)
        elif encoder_type == 'radar_encoder_temporal':
            from src.encoders.radar_encoder_temporal import RadarEncoderTemporal
            return RadarEncoderTemporal(**radar_cfg)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def _load_encoder_weights(self, encoder, weights_path: str) -> None:
        """Load weights into encoder."""
        weights = torch.load(weights_path, map_location='cpu')

        if not isinstance(weights, dict):
            raise ValueError(f"Weights must be dict, got {type(weights)}")

        encoder.load_state_dict(weights, strict=True)

    def _freeze_encoder_parameters(self, encoder) -> None:
        """Freeze all encoder parameters."""
        for p in encoder.parameters():
            p.requires_grad = False

    def _update_encoder_dimensions(self, encoder) -> None:
        """Update model dimensions based on encoder."""
        if not hasattr(encoder, 'embed_dim'):
            raise ValueError("Encoder must have embed_dim attribute")

        actual_embed_dim = encoder.embed_dim
        self._actual_embed_dim = actual_embed_dim

        if actual_embed_dim != self.mmwave_latent_dim:
            self._rebuild_projection_layers(actual_embed_dim)

    def _log_encoder_loading_success(self, encoder_version_path: str, radar_cfg: Dict[str, Any]) -> None:
        """Log successful encoder loading."""
        encoder_type = radar_cfg.get('encoder_type', 'unknown')
        embed_dim = self._actual_embed_dim
        max_seq_len = radar_cfg.get('max_sequence_length', 'N/A')

        print(f"✅ Loaded radar encoder: {encoder_type}")
        print(f"   Path: {encoder_version_path}")
        print(f"   Embed dim: {embed_dim}")
        print(f"   Max sequence length: {max_seq_len}")

    def _wrap_encoder_for_training(self, encoder):
        """
        Wrap radar encoder to handle training pipeline input format.

        Converts input_wave_embeds [B, N, C] to separate views expected by radar encoders.
        """
        class EncoderWrapper(nn.Module):
            """Simple wrapper for radar encoder to handle training pipeline input format."""

            def __init__(self, base_encoder):
                super().__init__()
                self.base_encoder = base_encoder
                self.embed_dim = base_encoder.embed_dim
                self._encoder_config = getattr(base_encoder, '_encoder_config', {})

            def _convert_to_views(self, input_wave_embeds: torch.Tensor):
                """Convert input tensor to three separate views."""
                batch_size, n_points, channels = input_wave_embeds.shape

                # Determine expected points per view from config
                encoder_type = self._encoder_config.get('encoder_type', 'radar_encoder_temporal')
                max_seq_len = self._encoder_config.get('max_sequence_length', 248)

                if encoder_type == 'radar_encoder_vit' and n_points == max_seq_len * 3:
                    # Standard three-view format
                    points_per_view = max_seq_len
                else:
                    # Equal splitting
                    if n_points % 3 != 0:
                        raise ValueError(f"Input points {n_points} must be divisible by 3 for three views")
                    points_per_view = n_points // 3

                # Split into three views and transpose for encoder format
                range_data = input_wave_embeds[:, :points_per_view, :].transpose(1, 2)
                doppler_data = input_wave_embeds[:, points_per_view:2*points_per_view, :].transpose(1, 2)
                azimuth_data = input_wave_embeds[:, 2*points_per_view:, :].transpose(1, 2)

                return range_data, doppler_data, azimuth_data

            def forward(self, input_wave_embeds: torch.Tensor, return_sequence: bool = False):
                """Forward pass with strict input validation."""
                if not isinstance(input_wave_embeds, torch.Tensor):
                    raise TypeError(f"input_wave_embeds must be torch.Tensor, got {type(input_wave_embeds)}")

                if input_wave_embeds.dim() != 3:
                    raise ValueError(f"input_wave_embeds must be 3D tensor, got shape {input_wave_embeds.shape}")

                # Convert to views
                range_data, doppler_data, azimuth_data = self._convert_to_views(input_wave_embeds)

                # Forward through base encoder
                return self.base_encoder(
                    range_data=range_data,
                    doppler_data=doppler_data,
                    azimuth_data=azimuth_data,
                    return_sequence=return_sequence
                )

        return EncoderWrapper(encoder).eval()


    def _rebuild_projection_layers(self, actual_embed_dim: int) -> None:
        """Rebuild projection layers with new embed dimension."""
        # Update dimensions
        self.mmwave_latent_dim = actual_embed_dim
        self.vae_latent_dim = actual_embed_dim

        # Rebuild projection layers
        self.mm_projection_layers = nn.Linear(actual_embed_dim, self.config.hidden_size)

        # Initialize weights
        nn.init.xavier_uniform_(self.mm_projection_layers.weight)
        if self.mm_projection_layers.bias is not None:
            nn.init.constant_(self.mm_projection_layers.bias, 0)

    def get_encoder_info(self) -> Dict[str, Any]:
        """Get information about the loaded encoder."""
        return {
            'embed_dim': self._actual_embed_dim or self.mmwave_latent_dim,
            'mmwave_latent_dim': self.mmwave_latent_dim,
            'has_encoder': hasattr(self, 'wave_encoder'),
            'encoder_config': self._encoder_config
        }

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
                wave_features = self.mm_projection_layers(self.wave_encoder(input_wave_embeds, return_sequence=True))

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