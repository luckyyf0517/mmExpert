"""
Vision Transformer radar encoder using the new abstraction layer.

This module provides a clean, extensible ViT-based radar encoder implementation
that integrates with the mmExpert framework's core abstractions.
Treats radar views as 2D grayscale images and processes them with Vision Transformer.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from ..core.base import BaseEncoder, ModalityData, ModalityType, EncodingResult
from ..core.config import EncoderConfig
from ..core.registry import register_encoder


class ViewEncoder(nn.Module):
    """Vision Transformer encoder for a single radar view."""

    def __init__(self, vit_model_name: str, patch_sz: Tuple[int, int],
                 target_sz: Tuple[int, int], freeze_bb: bool, max_seq_len: int,
                 use_repeat_in_chans: bool = True):
        super().__init__()

        # Handle rectangular patches properly
        if isinstance(patch_sz, (list, tuple)):
            # Handle nested tuples/lists by flattening
            if len(patch_sz) == 2 and isinstance(patch_sz[0], (list, tuple)):
                # This is a nested structure like ([32, 16], [32, 16])
                # Extract the first pair as the patch size
                patch_sz = list(patch_sz[0]) if isinstance(patch_sz[0], tuple) else patch_sz[0]

            # Ensure we have a simple list/tuple of 2 integers
            if isinstance(patch_sz, (list, tuple)) and len(patch_sz) == 2 and all(isinstance(p, int) for p in patch_sz):
                self.patch_size = list(patch_sz)  # Store as simple list [height, width]
            else:
                raise ValueError(f"Invalid patch_sz format after processing: {patch_sz}. Expected [height, width] with integer values.")
        else:
            # For square patches, convert to list format
            self.patch_size = [patch_sz, patch_sz]  # Store as list for consistency

        # Create Vision Transformer model with rectangular patches
        # timm supports rectangular patches by setting patch_size as (height, width)
        self.vit = timm.create_model(
            vit_model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=target_sz,
            patch_size=tuple(self.patch_size),  # Use rectangular patch size
            in_chans=1 if not use_repeat_in_chans else 3  # Set input channels based on use_repeat_in_chans
        )

        self.use_repeat_in_chans = use_repeat_in_chans
        self.target_size = target_sz
        self.embed_dim = self.vit.embed_dim
        self.max_sequence_length = max_seq_len  # Store max sequence length

        # Freeze backbone if requested
        if freeze_bb:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Calculate expected patches using original rectangular patch sizes
        # This represents the intended token count based on radar data dimensions
        # At this point, self.patch_size is guaranteed to be a list of 2 integers
        expected_patches = (target_sz[0] // self.patch_size[0]) * (target_sz[1] // self.patch_size[1])

        # Verify that max_sequence_length matches the expected number of patches (excluding CLS)
        if max_seq_len != expected_patches:
            raise ValueError(
                f"max_sequence_length ({max_seq_len}) does not match expected number of patches ({expected_patches}). "
                f"Target size: {target_sz}, Patch size: {self.patch_size}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for view encoder.

        Args:
            x: Input tensor [batch_size, channels, height, width]

        Returns:
            Encoded features [batch_size, num_patches, embed_dim]
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(1)  # Add batch and channel dimensions

        assert x.size(2) == self.target_size[0] and x.size(3) == self.target_size[1], \
            f"Input size {x.size(2)}x{x.size(3)} does not match target size {self.target_size[0]}x{self.target_size[1]}"

        # Ensure correct number of input channels
        if x.size(1) == 1 and self.use_repeat_in_chans:
            x = x.repeat(1, 3, 1, 1)  # Replicate grayscale to 3 channels only if use_repeat_in_chans is True

        # Extract features using ViT forward_features to get patch tokens (includes CLS token)
        features = self.vit.forward_features(x)  # [batch_size, num_patches+1, embed_dim]

        return features


@register_encoder(
    name="radar_encoder_vit",
    description="Vision Transformer radar encoder with adaptive patch sizes for different views",
    tags=["radar", "multimodal", "encoder", "vit", "vision-transformer"]
)
class RadarEncoderViT(BaseEncoder):
    """
    Vision Transformer radar encoder for processing mmWave radar data.

    This encoder supports:
    - Multi-view radar processing (range-time, doppler-time, azimuth-time)
    - Adaptive patch sizes for different view dimensions
    - Vision Transformer backbone with configurable variants
    - Sequence-level encoding
    - Flexible input handling
    - Patch-based tokenization of radar spectrograms
    """

    # Mapping from radar_views config string to view names
    RADAR_VIEW_MAP = {
        'all': ['range_time', 'doppler_time', 'azimuth_time'],
        'doppler': ['doppler_time'],
        'udoppler': ['doppler_time'],
        'range': ['range_time'],
        'azimuth': ['azimuth_time'],
    }

    def __init__(self,
                 embed_dim: int = 512,
                 input_dims: Dict[str, int] = None,
                 dropout: float = 0.1,
                 vit_model: str = "vit_base_patch16_224",
                 freeze_backbone: bool = False,
                 use_layer_norm: bool = True,
                 use_repeat_in_chans: bool = True,
                 pool_type: str = "avg",  # "avg" (default, excludes CLS token), "token", "max"
                 patch_size_range: list = [32, 16], # [height, width] for range view
                 patch_size_other: list = [16, 16], # [height, width] for other views
                 max_sequence_length: int = 248,  # Matches: 256×496 with patch size [256,2] or 128×496 with patch size [128,2]
                 radar_views: str = 'all',  # Which radar views to use: 'all', 'doppler'/'udoppler', 'range', 'azimuth'
                 **kwargs):
        """
        Initialize Vision Transformer radar encoder.

        Args:
            embed_dim: Final embedding dimension
            input_dims: Input dimensions for each radar view
            dropout: Dropout rate
            vit_model: Name of the ViT model from timm
            freeze_backbone: Whether to freeze the ViT backbone
            use_layer_norm: Whether to use layer normalization
            use_repeat_in_chans: Whether to repeat single-channel input to 3 channels (True) or use 1-channel ViT (False)
            pool_type: Type of pooling to use ("avg" (default, excludes CLS token), "token", "max").
            patch_size_range: Patch size for range-time view [height, width] - rectangular for 256×T input
            patch_size_other: Patch size for doppler/azimuth views [height, width] - square for 128×T input
            max_sequence_length: Maximum sequence length for positional encoding
            **kwargs: Additional parameters (ignored, for compatibility)
        """
        super().__init__(embed_dim=embed_dim, modality=ModalityType.RADAR)

        if not TIMM_AVAILABLE:
            raise ImportError("timm package is required for RadarEncoderViT. Please install with: pip install timm")

        # Resolve selected views from config
        self.radar_views = radar_views
        selected_views = self.RADAR_VIEW_MAP.get(radar_views, self.RADAR_VIEW_MAP['all'])

        # Default input dimensions if not provided
        if input_dims is None:
            input_dims = {
                "range_time": 256,
                "doppler_time": 128,
                "azimuth_time": 128
            }

        # Filter input_dims to only include selected views
        input_dims = {k: v for k, v in input_dims.items() if k in selected_views}
        self.input_dims = input_dims

        # Canonical view order for stable concatenation. The fusion layer
        # weights are partitioned by this order during training, so later uses
        # must preserve it even if YAML serialization reorders input_dims.
        self.canonical_view_order = list(selected_views)
        self.dropout = dropout
        self.vit_model = vit_model
        self.freeze_backbone = freeze_backbone
        self.use_layer_norm = use_layer_norm
        self.use_repeat_in_chans = use_repeat_in_chans
        self.pool_type = pool_type
        self.patch_size_range = patch_size_range
        self.patch_size_other = patch_size_other
        self.max_sequence_length = max_sequence_length

        # Create Vision Transformer encoders for each view
        self.view_encoders = nn.ModuleDict()
        for view_name, input_dim in input_dims.items():
            if view_name == "range_time":
                patch_size = patch_size_range
            else:
                patch_size = patch_size_other
            target_size = (int(input_dim), 496)

            self.view_encoders[view_name] = self._create_view_encoder(
                vit_model, patch_size, target_size, freeze_backbone, max_sequence_length, use_repeat_in_chans
            )

        # Get ViT feature dimension from the first encoder
        sample_encoder = list(self.view_encoders.values())[0]
        vit_embed_dim = sample_encoder.embed_dim

        # Ensure vit_embed_dim is an integer
        if not isinstance(vit_embed_dim, int):
            raise TypeError(f"vit_embed_dim must be int, got {type(vit_embed_dim)}: {vit_embed_dim}")

        # Fusion layer to combine multi-view features
        total_views = len(input_dims)

        # Ensure embed_dim is an integer
        if not isinstance(embed_dim, int):
            raise TypeError(f"embed_dim must be int, got {type(embed_dim)}: {embed_dim}")

        self.fusion_layer = nn.Linear(vit_embed_dim * total_views, embed_dim)

        # REMOVED: sequence_projection layer - was causing random seed dependency issues
        # The layer was never actually trained due to torch.no_grad() in trainer.py
        # This created a fragile dependency on random seed alignment

        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = nn.Identity()

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Sequence support
        self._supports_sequence = True

        # Initialize parameters
        self._initialize_parameters()

    # REMOVED: _ensure_sequence_projection method - no longer needed
    # This method was creating a random layer that was never trained due to torch.no_grad()
    # The resulting random seed dependency made the system unreliable

    def _create_view_encoder(self, vit_model: str, patch_size,
                           target_size: Tuple[int, int], freeze_backbone: bool,
                           max_sequence_length: int, use_repeat_in_chans: bool = True) -> nn.Module:
        """Create Vision Transformer encoder for a single radar view."""
        # Convert patch_size to tuple if it's a list
        if isinstance(patch_size, list):
            patch_size_tuple = tuple(patch_size)
        elif isinstance(patch_size, tuple):
            patch_size_tuple = patch_size
        else:
            # Single value, create square patch
            patch_size_tuple = (patch_size, patch_size)
        return ViewEncoder(vit_model, patch_size_tuple, target_size, freeze_backbone, max_sequence_length, use_repeat_in_chans)

    def _initialize_parameters(self) -> None:
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode(self,
               data: ModalityData,
               return_sequence: bool = False,
               **kwargs) -> EncodingResult:
        """
        Encode radar data using Vision Transformer.

        Args:
            data: Radar modality data
            return_sequence: Whether to return sequence-level features
            **kwargs: Additional encoding arguments

        Returns:
            Encoding result with features and metadata
        """
        # Extract radar data
        radar_data = data.data
        first_encoder = None

        # Handle different input formats
        if isinstance(radar_data, dict):
            # Multi-view format
            features = self._encode_multi_view(radar_data)
            # Get first encoder for potential use in pooling
            if self.view_encoders:
                first_encoder = next(iter(self.view_encoders.values()))
        elif isinstance(radar_data, torch.Tensor):
            # Single tensor format (assume it's already processed)
            features = radar_data
        else:
            raise ValueError(f"Unsupported radar data format: {type(radar_data)}")

        # Apply layer normalization and dropout
        features = self.layer_norm(features)
        features = self.dropout_layer(features)

        # Apply pooling using timm's pool method
        pooled_features = first_encoder.vit.pool(features, pool_type=self.pool_type)

        # Apply sequence projection if needed
        if return_sequence and features.dim() == 3:
            # Exclude CLS token for sequence features (only patch tokens)
            # Use patch_features directly as sequence_features (no random projection)
            sequence_features = features[:, 1:]  # Remove CLS token
        else:
            sequence_features = features

        # Create encoding result
        if return_sequence and sequence_features.dim() == 3:
            # Return sequence features
            return EncodingResult(
                features=pooled_features,
                sequence_features=sequence_features,
                metadata={
                    "modality": "radar",
                    "sequence_length": sequence_features.size(1),
                    "embed_dim": self.embed_dim,
                    "model_type": "vit",
                    "vit_model": self.vit_model,
                    "pool_type": self.pool_type
                }
            )
        else:
            # Return pooled features
            return EncodingResult(
                features=pooled_features,
                metadata={
                    "modality": "radar",
                    "embed_dim": self.embed_dim,
                    "model_type": "vit",
                    "vit_model": self.vit_model,
                    "pool_type": self.pool_type
                }
            )

    def _encode_multi_view(self, radar_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multi-view radar data using Vision Transformers.

        Args:
            radar_data: Dictionary with radar views (only configured views are expected)

        Returns:
            Encoded features
        """
        view_features = []
        first_encoder = None

        # Process each configured view in canonical order.
        # Iterating self.view_encoders.items() would follow ModuleDict insertion order,
        # which depends on input_dims dict ordering (can be reordered by YAML
        # serialization).  Using canonical_view_order keeps concatenation stable
        # so the fusion layer weights always see [range, doppler, azimuth].
        for view_name in self.canonical_view_order:
            if view_name not in self.view_encoders:
                continue
            encoder = self.view_encoders[view_name]
            if view_name not in radar_data or radar_data[view_name] is None:
                raise ValueError(
                    f"Configured radar view '{view_name}' is missing from input data. "
                    f"Available keys: {list(radar_data.keys())}. "
                    f"Configured views (radar_views={self.radar_views}): {list(self.view_encoders.keys())}"
                )

            view_data = radar_data[view_name]

            # Ensure data is in the right format for ViT
            # Input should be [batch, height, width] for grayscale
            if view_data.dim() == 3:
                # [batch, D, T] - treat as [batch, height, width]
                pass
            elif view_data.dim() == 2:
                # Add batch dimension
                view_data = view_data.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected tensor dimensions for {view_name}: {view_data.dim()}")

            # Store first encoder for potential use in pooling
            if first_encoder is None:
                first_encoder = encoder

            # Encode view using ViT
            encoded_view = encoder(view_data)  # [batch, num_patches, embed_dim]
            view_features.append(encoded_view)

        # Concatenate view features along feature dimension
        if len(view_features) > 1:
            concatenated = torch.cat(view_features, dim=-1)  # [batch, seq_len, embed_dim * num_views]
            features = self.fusion_layer(concatenated)  # [batch, seq_len, embed_dim]
        else:
            features = view_features[0]
            # Project to target embed_dim if needed
            if features.size(-1) != self.embed_dim:
                features = self.fusion_layer(features)

        return features

    def _get_features_and_pool(self, features: torch.Tensor, encoder=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method to get both sequence features and pooled features.

        Args:
            features: Input features [batch_size, seq_len, embed_dim]
            encoder: The encoder to use for pooling (if needed)

        Returns:
            Tuple of (pooled_features, sequence_features)
        """
        # Pool features based on pool_type
        pooled_features = self._apply_pooling(features, encoder)

        return pooled_features, features

    def _get_batch_size(self, radar_data: Dict[str, torch.Tensor]) -> int:
        """Get batch size from radar data."""
        for view_data in radar_data.values():
            if view_data is not None:
                return view_data.size(0)
        return 1  # Default batch size

    def forward(self, range_data=None, doppler_data=None, azimuth_data=None, return_sequence=False):
        """
        Forward pass for backward compatibility.

        Args:
            range_data: Range-time data [batch, 256, T]
            doppler_data: Doppler-time data [batch, 128, T]
            azimuth_data: Azimuth-time data [batch, 128, T]
            return_sequence: Whether to return sequence features

        Returns:
            Encoded features
        """
        # Create modality data dictionary
        radar_dict = {}
        if range_data is not None:
            radar_dict["range_time"] = range_data
        if doppler_data is not None:
            radar_dict["doppler_time"] = doppler_data
        if azimuth_data is not None:
            radar_dict["azimuth_time"] = azimuth_data

        # Remove None values
        radar_dict = {k: v for k, v in radar_dict.items() if v is not None}

        # Create ModalityData object
        modality_data = ModalityData(
            data=radar_dict,
            modality=ModalityType.RADAR,
            metadata={"format": "multi_view", "encoder_type": "vit"}
        )

        # Encode
        result = self.encode(modality_data, return_sequence=return_sequence)

        if return_sequence:
            return result.sequence_features
        else:
            return result.features

    def unfreeze_last_layers(self, num_layers: int = 2):
        """
        Unfreeze the last few layers of the ViT backbone for fine-tuning.

        Args:
            num_layers: Number of last layers to unfreeze
        """
        for encoder in self.view_encoders.values():
            vit = encoder.vit
            total_blocks = len(vit.blocks)

            # Unfreeze the last few transformer blocks
            for i in range(total_blocks - num_layers, total_blocks):
                for param in vit.blocks[i].parameters():
                    param.requires_grad = True

            # Always unfreeze the classification head (if exists)
            if hasattr(vit, 'head') and vit.head is not None:
                for param in vit.head.parameters():
                    param.requires_grad = True


# Factory function for creating ViT radar encoder
def create_radar_encoder_vit(config: EncoderConfig) -> RadarEncoderViT:
    """Create Vision Transformer radar encoder from configuration."""
    return RadarEncoderViT(
        embed_dim=config.embed_dim,
        input_dims=config.get("input_dims"),
        dropout=config.dropout,
        vit_model=config.get("vit_model", "vit_base_patch16_224"),
        freeze_backbone=config.get("freeze_backbone", False),
        use_layer_norm=config.get("use_layer_norm", True),
        use_repeat_in_chans=config.get("use_repeat_in_chans", True),
        pool_type=config.get("pool_type", "avg"),  # Default to "avg" pooling
        patch_size_range=config.get("patch_size_range", [32, 16]),
        patch_size_other=config.get("patch_size_other", [16, 16]),
        max_sequence_length=config.get("max_sequence_length", 248),
        radar_views=config.get("radar_views", "all")
    )


# Component is automatically registered via @register_encoder decorator
