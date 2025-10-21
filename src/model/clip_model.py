import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from easydict import EasyDict as edict

from .clip_transformer import Transformer, build_attention_mask, LayerNorm
from .clip_text_encoder import TextEncoder
from .clip_radar_encoder import RadarEncoder
from .clip_loss import create_loss

# Suppress specific warning from transformers
import warnings
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.")


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
            # Improved initialization for SigLIP
            # Logit scale: smaller initial value for better stability
            # Based on SigLIP paper, logit_scale should start small and learn
            init_logit_scale = np.log(1 / 0.07)  # Similar to CLIP's temperature
            self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

            # Logit bias: initialize to 0 or slightly negative for better sigmoid activation
            # Starting with 0 allows the model to learn the optimal bias
            self.logit_bias = nn.Parameter(torch.tensor(0.0))
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

        # Training step counter for adaptive normalization
        self._training_step = 0
        self._log_norm_stats = False  # Set to True for debugging normalization

        self.initialize_parameters()
        self.configure_optimizers()

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

        # Improved feature normalization with numerical stability
        radar_features = self._normalize_features(radar_features, feature_type="radar")
        text_features = self._normalize_features(text_features, feature_type="text")
        return radar_features, text_features

    def _normalize_features(self, features: torch.Tensor, feature_type: str = "generic", eps: float = 1e-8) -> torch.Tensor:
        """
        Improved feature normalization with numerical stability and SigLIP compatibility.

        Args:
            features: Input features [b, embed_dim]
            feature_type: Type of features ("radar", "text", "generic") for debugging
            eps: Small constant for numerical stability

        Returns:
            Normalized features
        """
        if features is None:
            raise ValueError(f"Cannot normalize None features for {feature_type}")

        # Ensure features are 2D [batch_size, feature_dim]
        if features.dim() != 2:
            raise ValueError(f"Expected 2D features for {feature_type}, got {features.dim()}D")

        # Compute L2 norm with numerical stability
        features_norm = torch.norm(features, p=2, dim=1, keepdim=True)
        features_norm = torch.clamp(features_norm, min=eps)  # Prevent division by zero

        # Basic L2 normalization
        normalized_features = features / features_norm

        # Additional normalization strategies for SigLIP vs standard CLIP
        if self.use_siglip:
            # For SigLIP, we can use slightly more aggressive normalization
            # to improve training stability with sigmoid loss
            if hasattr(self, '_training_step') and self._training_step < 1000:
                # Early training: use adaptive normalization
                # This helps with convergence in early stages
                adaptive_scale = min(1.0, self._training_step / 1000.0)
                normalized_features = adaptive_scale * normalized_features + (1 - adaptive_scale) * features / features_norm
        else:
            # For standard CLIP, strict L2 normalization is typically better
            pass

        # Optional: Log normalization statistics for debugging
        if hasattr(self, '_log_norm_stats') and self._log_norm_stats:
            with torch.no_grad():
                mean_norm = features_norm.mean().item()
                std_norm = features_norm.std().item()
                print(f"[{feature_type}] Feature norms - Mean: {mean_norm:.6f}, Std: {std_norm:.6f}")

        return normalized_features

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
        # Update training step counter
        self._training_step += 1

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