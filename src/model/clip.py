import os

# Set offline mode for Hugging Face to avoid network requests
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import json
import einops
import torchvision
import numpy as np
import pytorch_lightning as pl

from typing import Tuple, Union
from collections import OrderedDict
from einops.layers.torch import Rearrange   
from termcolor import colored
from easydict import EasyDict as edict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from timm.models.vision_transformer import _create_vision_transformer

import warnings
from src.model.clip_loss import create_loss

# Suppress specific warning from transformers
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.")

# Constants
DEFAULT_HF_CACHE_DIR = '/root/autodl-tmp/mmExpert/huggingface'


def build_attention_mask(context_length):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1) # zero out the lower diagonal
    return mask


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * mlp_ratio)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * mlp_ratio, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: int = 4, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, mlp_ratio, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class TextEncoder(nn.Module):
    def __init__(self, model_name: str, text_pooling: str = 'pooler', unfreeze_last_layer_num: int = 0, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.text_pooling = text_pooling
        self.unfreeze_last_layer_num = unfreeze_last_layer_num
        
        # Set cache directory for Hugging Face models
        cache_dir = DEFAULT_HF_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load tokenizer and model with offline mode
        try:
            # Set environment variables for offline mode
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=True,  # Only use local files
                use_fast=True
            )
            self.text_encoder = AutoModel.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=True  # Only use local files
            )
        except Exception as e:
            print(f"Error loading model {model_name} in offline mode: {e}")
            print("Trying to load with network access...")
            # Fallback to online mode if offline fails
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                self.text_encoder = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            except Exception as e2:
                print(f"Error loading model {model_name} with network: {e2}")
                raise
        self.text_encoder.eval()
        for name, param in self.text_encoder.named_parameters():
            # Handle different model architectures
            if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layer'):
                # BERT-style models
                num_layers = len(self.text_encoder.encoder.layer)
                unfreeze_param = False
                for i in range(self.unfreeze_last_layer_num):
                    if 'layer.%d' % (num_layers - i) in name:
                        unfreeze_param = True
                    if 'pooler' in name:
                        unfreeze_param = True
            elif hasattr(self.text_encoder, 'layers') or hasattr(self.text_encoder, 'model'):
                # Handle other architectures (like Phi-3)
                unfreeze_param = False
                # For simplicity, unfreeze all parameters for these models
                if self.unfreeze_last_layer_num > 0:
                    unfreeze_param = True
            else:
                # Default case
                unfreeze_param = False

            param.requires_grad = unfreeze_param
        
    def encode(self, text, device='cuda'):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        if self.text_pooling == 'mean': 
            out = outputs.last_hidden_state.mean(dim=1)
        elif self.text_pooling == 'pooler': 
            out = outputs.pooler_output
        elif self.text_pooling == 'max': 
            out = outputs.last_hidden_state.max(dim=1)[0]
        return out
    

class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str,
                 embed_dim: int,
                 image_resolution: int,
                 pretrained: bool = True,
                 **kwargs):
        super().__init__()
        if image_resolution == [496, 128]:
            self.window_size = None
        else:
            self.window_size = 128
            self.stride = 16
            self.unfold = nn.Sequential(
                nn.Unfold(kernel_size=(self.window_size, self.window_size), stride=self.stride),
                Rearrange('b (c h w) n -> (b n) c h w', c=1, h=self.window_size, w=self.window_size))

        self.model_name = model_name
        if 'vit' in model_name:
            self.visual = _create_vision_transformer(model_name, pretrained=pretrained, in_chans=1, img_size=image_resolution, num_classes=embed_dim, **kwargs)
        elif 'resnet' in model_name:
            self.visual = torchvision.models.resnet18(pretrained=pretrained)
            self.visual.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.visual.fc = nn.Linear(self.visual.fc.in_features, embed_dim)

    def forward(self, x):
        if self.window_size is not None:
            batch_size = x.size(0)
            x = self.unfold(x)
            x = self.visual(x)
            x = einops.rearrange(x, '(b n) c -> b n c', b=batch_size)
        else:
            x = self.visual(x)
        return x

    def encode_to_sequence(self, x):
        if self.window_size is not None:
            batch_size = x.size(0)
            x = self.unfold(x)
            x = self.visual(x)
            x = einops.rearrange(x, '(b n) c -> b n c', b=batch_size)
        else:
            if 'vit' in self.model_name:
                x = self.visual.forward_features(x)
                x = x[:, 1:, :] # [b, n, c]
            elif 'resnet' in self.model_name:
                x = self.visual.conv1(x)
                x = self.visual.bn1(x)
                x = self.visual.relu(x)
                x = self.visual.maxpool(x)
                x = self.visual.layer1(x)
                x = self.visual.layer2(x)
                x = self.visual.layer3(x)
                x = einops.rearrange(x, 'b c h w -> b (h w) c')
        return x


class RadarEncoder(nn.Module):
    """
    Parallel encoder for radar views: range_time, doppler_time, azimuth_time
    Supports single view or multiple views based on radar_views configuration.
    Each view has its own encoder and the features are combined.
    """
    def __init__(self,
                 model_name: str,
                 embed_dim: int,
                 # Resolutions for each radar view
                 range_resolution: Tuple[int, int] = (256, 496),  # (range_bins, time_frames)
                 doppler_resolution: Tuple[int, int] = (128, 496),  # (doppler_bins, time_frames)
                 azimuth_resolution: Tuple[int, int] = (128, 496),  # (azimuth_bins, time_frames)
                 pretrained: bool = True,
                 fusion_method: str = 'concat',  # 'concat', 'add', 'attention'
                 adaptive_patch_size: bool = False,  # Enable adaptive patch sizing for uniform sequence length
                 radar_views: str = 'all',  # 'all', 'doppler_only', 'range_only', 'azimuth_only'
                 **kwargs):
        super().__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.fusion_method = fusion_method
        self.adaptive_patch_size = adaptive_patch_size
        self.radar_views = radar_views

        # Determine which encoders to create based on radar_views configuration
        self.use_range = self.radar_views in ['all', 'range_only']
        self.use_doppler = self.radar_views in ['all', 'doppler_only']
        self.use_azimuth = self.radar_views in ['all', 'azimuth_only']

        num_views = sum([self.use_range, self.use_doppler, self.use_azimuth])

        if adaptive_patch_size:
            # Calculate optimal patch sizes for uniform sequence length
            patch_configs = self._calculate_adaptive_patch_sizes(
                range_resolution, doppler_resolution, azimuth_resolution
            )
            self.range_patch_size = patch_configs['range']
            self.doppler_patch_size = patch_configs['doppler']
            self.azimuth_patch_size = patch_configs['azimuth']
            print(f"[INFO] Adaptive patch sizes - Range: {self.range_patch_size}, "
                  f"Doppler: {self.doppler_patch_size}, Azimuth: {self.azimuth_patch_size}")

        # Create encoders based on configuration
        if self.use_range:
            self.range_encoder = self._create_encoder(model_name, embed_dim, range_resolution, pretrained, 'range', **kwargs)
        if self.use_doppler:
            self.doppler_encoder = self._create_encoder(model_name, embed_dim, doppler_resolution, pretrained, 'doppler', **kwargs)
        if self.use_azimuth:
            self.azimuth_encoder = self._create_encoder(model_name, embed_dim, azimuth_resolution, pretrained, 'azimuth', **kwargs)

        # Fusion layers
        if fusion_method == 'concat':
            # Concatenate features and project to embed_dim
            self.fusion_proj = nn.Linear(embed_dim * num_views, embed_dim)
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
            self.fusion_proj = nn.Linear(embed_dim, embed_dim)
        elif fusion_method == 'add':
            # Simple addition (no additional parameters needed)
            pass
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        print(f"[INFO] RadarEncoder initialized with views: {self.radar_views} (active views: {num_views})")

    def _calculate_adaptive_patch_sizes(self, range_res, doppler_res, azimuth_res):
        """
        Calculate optimal patch sizes for uniform sequence length across all three radar views.

        Strategy: Use specified patch sizes (32*16, 16*16, 16*16) to achieve consistent sequence length.
        """
        # Target patch sizes as specified
        range_patch_size = (32, 16)  # (height, width)
        doppler_patch_size = (16, 16)  # (height, width)
        azimuth_patch_size = (16, 16)  # (height, width)

        # Calculate sequence lengths to verify consistency
        range_h, range_w = range_res
        doppler_h, doppler_w = doppler_res
        azimuth_h, azimuth_w = azimuth_res

        # Ensure dimensions are divisible by patch sizes
        assert range_h % range_patch_size[0] == 0, f"Range height {range_h} not divisible by {range_patch_size[0]}"
        assert range_w % range_patch_size[1] == 0, f"Range width {range_w} not divisible by {range_patch_size[1]}"
        assert doppler_h % doppler_patch_size[0] == 0, f"Doppler height {doppler_h} not divisible by {doppler_patch_size[0]}"
        assert doppler_w % doppler_patch_size[1] == 0, f"Doppler width {doppler_w} not divisible by {doppler_patch_size[1]}"
        assert azimuth_h % azimuth_patch_size[0] == 0, f"Azimuth height {azimuth_h} not divisible by {azimuth_patch_size[0]}"
        assert azimuth_w % azimuth_patch_size[1] == 0, f"Azimuth width {azimuth_w} not divisible by {azimuth_patch_size[1]}"

        # Calculate sequence lengths
        range_seq_len = (range_h // range_patch_size[0]) * (range_w // range_patch_size[1])
        doppler_seq_len = (doppler_h // doppler_patch_size[0]) * (doppler_w // doppler_patch_size[1])
        azimuth_seq_len = (azimuth_h // azimuth_patch_size[0]) * (azimuth_w // azimuth_patch_size[1])

        print(f"[INFO] Sequence lengths - Range: {range_seq_len}, Doppler: {doppler_seq_len}, Azimuth: {azimuth_seq_len}")

        return {
            'range': range_patch_size,
            'doppler': doppler_patch_size,
            'azimuth': azimuth_patch_size
        }

    def _create_encoder(self, model_name: str, embed_dim: int, resolution: Tuple[int, int], pretrained: bool, view_type: str = 'range', **kwargs):
        """Create an encoder for a specific radar view."""
        if 'vit' in model_name:
            # Set custom patch size if adaptive patch sizing is enabled
            if self.adaptive_patch_size:
                if view_type == 'range':
                    patch_size = self.range_patch_size
                elif view_type == 'doppler':
                    patch_size = self.doppler_patch_size
                elif view_type == 'azimuth':
                    patch_size = self.azimuth_patch_size
                else:
                    raise ValueError(f"Unknown view type: {view_type}")

                kwargs['patch_size'] = patch_size
                print(f"[INFO] Creating {view_type} encoder with patch_size={patch_size} for resolution={resolution}")

            # For ViT, treat as 1-channel image with height=range_bins, width=time_frames
            # Use global_pool=True to get features instead of classification
            encoder = _create_vision_transformer(
                model_name,
                pretrained=pretrained,
                in_chans=1,
                img_size=resolution,
                num_classes=0,  # Set to 0 to get features without classification head
                **kwargs
            )
            # Add a projection layer to get the desired embed_dim
            encoder.proj = nn.Linear(encoder.embed_dim, embed_dim)
        elif 'resnet' in model_name:
            # For ResNet, adapt to 1-channel input
            encoder = torchvision.models.resnet18(pretrained=pretrained)
            encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            encoder.fc = nn.Linear(encoder.fc.in_features, embed_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        return encoder

    def encode_view(self, encoder, x):
        """Encode a single radar view."""
        if 'vit' in self.model_name:
            # For ViT, get features and project to embed_dim
            features = encoder.forward_features(x)  # [b, n+1, c] where n is sequence length
            features = features[:, 1:, :]  # Remove CLS token: [b, n, c]
            features = encoder.proj(features)  # Project to desired embed_dim
        elif 'resnet' in self.model_name:
            # For ResNet, get intermediate features
            x = encoder.conv1(x)
            x = encoder.bn1(x)
            x = encoder.relu(x)
            x = encoder.maxpool(x)
            x = encoder.layer1(x)
            x = encoder.layer2(x)
            x = encoder.layer3(x)
            # Reshape to sequence format: [b, h*w, c]
            features = einops.rearrange(x, 'b c h w -> b (h w) c')

        return features

    def forward(self, range_data, doppler_data, azimuth_data):
        """
        Forward pass for radar encoders with configurable views and None support.

        Args:
            range_data: [b, 256, T] range-time spectrum or None
            doppler_data: [b, 128, T] doppler-time spectrum or None
            azimuth_data: [b, 128, T] azimuth-time spectrum or None

        Returns:
            fused_features: [b, n, embed_dim] fused radar features
        """
        features_list = []

        # Encode each view based on configuration and data availability
        if self.use_range and range_data is not None:
            range_data_2d = range_data.unsqueeze(1)    # [b, 1, 256, T]
            range_features = self.encode_view(self.range_encoder, range_data_2d)  # [b, n_range, embed_dim]
            features_list.append(range_features)

        if self.use_doppler and doppler_data is not None:
            doppler_data_2d = doppler_data.unsqueeze(1)  # [b, 1, 128, T]
            doppler_features = self.encode_view(self.doppler_encoder, doppler_data_2d)  # [b, n_doppler, embed_dim]
            features_list.append(doppler_features)

        if self.use_azimuth and azimuth_data is not None:
            azimuth_data_2d = azimuth_data.unsqueeze(1)  # [b, 1, 128, T]
            azimuth_features = self.encode_view(self.azimuth_encoder, azimuth_data_2d)  # [b, n_azimuth, embed_dim]
            features_list.append(azimuth_features)

        # Check if we have any features
        if len(features_list) == 0:
            raise ValueError("No valid radar data provided (all inputs are None)")

        # Fuse features
        fused_features = self._fuse_features(features_list)

        return fused_features

    def _fuse_features(self, features_list):
        """Fuse features from configured radar views."""
        if len(features_list) == 1:
            # Single view mode - no fusion needed
            return features_list[0]

        if self.fusion_method == 'concat':
            if self.adaptive_patch_size:
                # With adaptive patch sizing, features have same sequence length
                # Stack along feature dimension: [b, n, n_views, embed_dim] -> [b, n, embed_dim*n_views]
                stacked = torch.stack(features_list, dim=2)
                b, n, n_views, d = stacked.shape
                fused = stacked.view(b, n, n_views * d)  # [b, n, embed_dim*n_views]
                fused = self.fusion_proj(fused)  # [b, n, embed_dim]
            else:
                # Original behavior for backward compatibility
                fused = torch.cat(features_list, dim=1)  # [b, n_total, embed_dim*n_views]
                fused = self.fusion_proj(fused)  # [b, n_total, embed_dim]

        elif self.fusion_method == 'add':
            # Simple addition (pad to same length first)
            max_len = max(feat.size(1) for feat in features_list)

            # Pad sequences to same length
            padded_features = []
            for feat in features_list:
                padded = F.pad(feat, (0, 0, 0, max_len - feat.size(1)))
                padded_features.append(padded)

            fused = sum(padded_features) / len(padded_features)  # [b, max_len, embed_dim]

        elif self.fusion_method == 'attention':
            # Stack features for attention: [b, n_views, n, embed_dim]
            max_len = max(feat.size(1) for feat in features_list)

            # Pad and stack
            padded_features = []
            for feat in features_list:
                padded = F.pad(feat, (0, 0, 0, max_len - feat.size(1)))
                padded_features.append(padded)

            stacked = torch.stack(padded_features, dim=1)  # [b, n_views, max_len, embed_dim]

            # Reshape for attention: [b, max_len, n_views, embed_dim] -> [b*max_len, n_views, embed_dim]
            b, n_views, n_seq, feat_dim = stacked.shape
            reshaped = stacked.view(b * n_seq, n_views, feat_dim)

            # Apply attention
            attended, _ = self.attention(reshaped, reshaped, reshaped)  # [b*max_len, n_views, embed_dim]

            # Reshape back and average across views
            attended = attended.view(b, n_seq, n_views, feat_dim)  # [b, max_len, n_views, embed_dim]
            fused = attended.mean(dim=2)  # [b, max_len, embed_dim]
            fused = self.fusion_proj(fused)  # [b, max_len, embed_dim]

        return fused

    def encode_to_sequence(self, range_data, doppler_data, azimuth_data):
        """Encode radar data to sequence format (same as forward)."""
        return self.forward(range_data, doppler_data, azimuth_data)
            

class CLIP(pl.LightningModule):
    """
    CLIP model for radar data with parallel encoders.
    - Uses three parallel encoders for range_time, doppler_time, azimuth_time views
    - The transformer processes the fused radar features
    """
    def __init__(self,
                 encoder_cfg: dict,
                 text_cfg: dict,
                 # transformer cfg
                 context_length: int,
                 transformer_width: int,
                 transformer_layers: int,
                 transformer_heads: int,
                 # training cfg
                 temperature: float,
                 use_siglip: bool = False,
                 learning_rate: float = 1.0e-04,
                 max_epochs: int = 50):
        super().__init__()

        # Radar encoder
        self.radar_encoder = RadarEncoder(**encoder_cfg)
        encoder_embed_dim = encoder_cfg['embed_dim']

        self.text_encoder = TextEncoder(**text_cfg)
        # Get actual embed_dim from the text encoder
        text_embed_dim = text_cfg.get('embed_dim', 3072)  # Default fallback
        self.text_projection = nn.Linear(text_embed_dim, encoder_embed_dim)

        # Transformer setup
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=build_attention_mask(context_length)
        )
        scale = transformer_width ** -0.5
        self.eot_token = nn.Parameter(scale * torch.randn(transformer_width))
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.use_siglip = use_siglip
        if use_siglip:
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        else:
            self.logit_scale = 1 / temperature

        loss_args = edict({
            'gather_with_grad': True,
            'rank': int(os.environ.get('RANK', 0)),
            'world_size': int(os.environ.get('WORLD_SIZE', 1)),
            'horovod': False,
            'local_loss': False,
            'siglip': use_siglip,
            'loss_dist_impl': 'bidir',
        })
        self.loss_fn = create_loss(loss_args)

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.initialize_parameters()
        self.configure_optimizers()
        
    # def state_dict(self, *, destination=None, prefix='', keep_vars=False):
    #     state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    #     # do not save text encoder
    #     for k in list(state_dict.keys()): 
    #         if 'text_encoder' in k: 
    #             state_dict.pop(k)
    #     return state_dict

    def initialize_parameters(self):
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=0.01)

        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    @property
    def dtype(self):
        # Get dtype from the first parameter of radar encoder
        return next(self.radar_encoder.parameters()).dtype

    
    def encode_radar(self, range_data, doppler_data, azimuth_data):
        """
        Encode radar data using parallel encoders.

        Args:
            range_data: [b, 256, T] range-time spectrum
            doppler_data: [b, 128, T] doppler-time spectrum
            azimuth_data: [b, 128, T] azimuth-time spectrum

        Returns:
            features: [b, embed_dim] encoded radar features
        """
        x = self.radar_encoder(range_data, doppler_data, azimuth_data)  # [b, n, embed_dim]

        # Apply transformer
        x = torch.cat([x, self.eot_token.to(x.dtype) + torch.zeros_like(x[:, :1, :])], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.transformer(x)
        x = self.ln_final(x[:, -1, :])  # Use EOT token representation

        return x

    def encode_text(self, text, device='cuda'):
        x = self.text_encoder.encode(text, device=device)
        x = self.text_projection(x)
        return x

    def forward(self, radar_data, text):
        """
        Forward pass for CLIP with radar data.

        Args:
            radar_data: Dict with keys 'range_time', 'doppler_time', 'azimuth_time'
            text: List of text captions

        Returns:
            radar_features: [b, embed_dim] normalized radar features
            text_features: [b, embed_dim] normalized text features
        """
        radar_features = self.encode_radar(
            radar_data['range_time'],
            radar_data['doppler_time'],
            radar_data['azimuth_time']
        )

        # Get device from the first non-None radar data
        device = None
        if radar_data['range_time'] is not None:
            device = radar_data['range_time'].device
        elif radar_data['doppler_time'] is not None:
            device = radar_data['doppler_time'].device
        elif radar_data['azimuth_time'] is not None:
            device = radar_data['azimuth_time'].device
        else:
            raise ValueError("All radar data is None, cannot determine device")

        text_features = self.encode_text(text, device=device)

        # normalized features
        radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return radar_features, text_features

    
    def compute_loss(self, batch):
        radar_features = batch['radar_features'] if 'radar_features' in batch else batch['image_features']
        text_features = batch['text_features']
        if not self.use_siglip:
            loss_clip = self.loss_fn(radar_features, text_features, logit_scale=self.logit_scale)
        else:
            loss_clip = self.loss_fn(radar_features, text_features, logit_scale=self.logit_scale, logit_bias=self.logit_bias)
        return {'loss_clip': loss_clip}

    def shared_step(self, batch, batch_idx, phase='train'):
        # Determine batch size from non-None radar data
        if batch['input_wave_range'] is not None:
            self.batch_size = batch['input_wave_range'].size(0)
        elif batch['input_wave_doppler'] is not None:
            self.batch_size = batch['input_wave_doppler'].size(0)
        elif batch['input_wave_azimuth'] is not None:
            self.batch_size = batch['input_wave_azimuth'].size(0)
        else:
            raise ValueError("All radar views are None")

        # Handle radar data with None support
        radar_data = {
            'range_time': batch['input_wave_range'],    # Could be None
            'doppler_time': batch['input_wave_doppler'],  # Could be None
            'azimuth_time': batch['input_wave_azimuth']   # Could be None
        }
        text = batch['caption']
        radar_features, text_features = self.forward(radar_data, text)
        batch['radar_features'] = radar_features
        batch['image_features'] = radar_features  # For compatibility with loss computation
        batch['text_features'] = text_features

        return batch
    
    def training_step(self, batch, batch_idx):
        batch = self.shared_step(batch, batch_idx, phase='train')
        loss_dict = self.compute_loss(batch)
        self.log_loss(loss_dict, phase='train')
        
        # Display loss in progress bar with more details
        self.log('train_loss', loss_dict['loss_clip'], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        
        # Also log learning rate for monitoring (safely)
        try:
            if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                self.log('lr', current_lr, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        except (AttributeError, IndexError):
            pass  # Skip if trainer or optimizers not available
        
        return loss_dict['loss_clip']
        
    def validation_step(self, batch, batch_idx):
        batch = self.shared_step(batch, batch_idx, phase='valid')
        loss_dict = self.compute_loss(batch)
        self.log_loss(loss_dict, phase='valid')
        
        # Display loss in progress bar
        self.log('val_loss', loss_dict['loss_clip'], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        
        return loss_dict['loss_clip']
    
    def configure_optimizers(self):
        lr = self.learning_rate

        opt = torch.optim.AdamW([
            {'params': self.text_encoder.parameters(), 'lr': lr / 2},
            {'params': self.text_projection.parameters(), 'lr': lr},
            {'params': self.radar_encoder.parameters(), 'lr': lr},
            {'params': self.transformer.parameters(), 'lr': lr},
        ], betas=(0.5, 0.9), weight_decay=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=0)
        return [opt], [scheduler]
    
    def log_loss(self, loss_dict, phase='train'): 
        for k, v in loss_dict.items(): 
            self.log(phase + '/' + k, v, on_step=self.training, on_epoch=not self.training, 
                     logger=True, batch_size=self.batch_size, rank_zero_only=True, sync_dist=False, add_dataloader_idx=False)
        del loss_dict
