"""
Vision Transformer radar encoder using the new abstraction layer.

This module provides a clean, extensible ViT-based radar encoder implementation
that integrates with the mmExpert framework's core abstractions.
Treats radar views as 2D grayscale images and processes them with Vision Transformer.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from ..core.base import BaseEncoder, ModalityData, ModalityType, EncodingResult
from ..core.config import EncoderConfig
from ..core.registry import register_encoder
from ..core.factory import auto_factory


class ViewEncoder(nn.Module):
    """Vision Transformer encoder for a single radar view."""
    
    def __init__(self, vit_model_name: str, patch_sz: Tuple[int, int],
                 target_sz: Tuple[int, int], freeze_bb: bool, max_seq_len: int):
        super().__init__()

        # Handle rectangular patches by using max dimension for ViT model
        # ViT requires square patches, so we'll use the larger dimension
        if isinstance(patch_sz, (list, tuple)):
            # Handle nested tuples/lists by flattening
            if len(patch_sz) == 2 and isinstance(patch_sz[0], (list, tuple)):
                # This is a nested structure like ([32, 16], [32, 16])
                # Extract the first pair as the patch size
                patch_sz = list(patch_sz[0]) if isinstance(patch_sz[0], tuple) else patch_sz[0]

            # Ensure we have a simple list/tuple of 2 integers
            if isinstance(patch_sz, (list, tuple)) and len(patch_sz) == 2 and all(isinstance(p, int) for p in patch_sz):
                # Use the larger dimension for ViT patch size
                patch_size_for_vit = max(patch_sz)
                self.patch_size = list(patch_sz)  # Store as simple list
            else:
                raise ValueError(f"Invalid patch_sz format after processing: {patch_sz}. Expected [height, width] with integer values.")
        else:
            # For square patches, use as-is
            patch_size_for_vit = patch_sz
            self.patch_size = [patch_sz, patch_sz]  # Store as list for consistency

        # Create Vision Transformer model with square patches
        self.vit = timm.create_model(
            vit_model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=target_sz,
            patch_size=patch_size_for_vit
        )

        self.patch_size_for_vit = patch_size_for_vit  # Store square patch size for ViT
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

        # Use adaptive average pooling to standardize number of patches
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.max_sequence_length)

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

        # Resize to target size
        x = torch.nn.functional.interpolate(
            x, size=self.target_size, mode='bilinear', align_corners=False
        )

        # Ensure 3-channel input for pretrained ViT (replicate grayscale)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Extract features using ViT
        features = self.vit(x)  # [batch_size, num_patches, embed_dim]

        # Standardize number of patches
        if features.dim() == 2:
            # If ViT returns 2D tensor [batch_size, embed_dim], reshape to 3D
            features = features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        elif features.size(1) != self.max_sequence_length:
            # Transpose for adaptive pooling
            features = features.transpose(1, 2)  # [batch_size, embed_dim, num_patches]
            features = self.adaptive_pool(features)  # [batch_size, embed_dim, target_patches]
            features = features.transpose(1, 2)  # [batch_size, target_patches, embed_dim]

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

    def __init__(self,
                 embed_dim: int = 512,
                 input_dims: Dict[str, int] = None,
                 dropout: float = 0.1,
                 vit_model: str = "vit_base_patch16_224",
                 freeze_backbone: bool = False,
                 use_layer_norm: bool = True,
                 patch_size_range: list = [32, 16], # [height, width] for range view
                 patch_size_other: list = [16, 16], # [height, width] for other views
                 max_sequence_length: int = 196,  # 14x14 patches for 224x224 input
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
            patch_size_range: Patch size for range-time view [height, width] - rectangular for 256×T input
            patch_size_other: Patch size for doppler/azimuth views [height, width] - square for 128×T input
            max_sequence_length: Maximum sequence length for positional encoding
            **kwargs: Additional parameters (ignored, for compatibility)
        """
        super().__init__(embed_dim=embed_dim, modality=ModalityType.RADAR)

        if not TIMM_AVAILABLE:
            raise ImportError("timm package is required for RadarEncoderViT. Please install with: pip install timm")

        # Default input dimensions if not provided
        if input_dims is None:
            input_dims = {
                "range_time": 256,
                "doppler_time": 128,
                "azimuth_time": 128
            }

        self.input_dims = input_dims
        self.dropout = dropout
        self.vit_model = vit_model
        self.freeze_backbone = freeze_backbone
        self.use_layer_norm = use_layer_norm
        self.patch_size_range = patch_size_range
        self.patch_size_other = patch_size_other
        self.max_sequence_length = max_sequence_length

        # Create Vision Transformer encoders for each view
        self.view_encoders = nn.ModuleDict()
        for view_name, input_dim in input_dims.items():
            if view_name == "range_time":
                patch_size = patch_size_range
                # Use original dimensions: 256 × T
                target_size = (256, 496)  # Original radar dimensions
            else:
                patch_size = patch_size_other
                # Use original dimensions: 128 × T
                target_size = (128, 496)  # Original radar dimensions

            self.view_encoders[view_name] = self._create_view_encoder(
                vit_model, patch_size, target_size, freeze_backbone, max_sequence_length
            )

        # Get ViT feature dimension from the first encoder
        sample_encoder = list(self.view_encoders.values())[0]
        vit_embed_dim = sample_encoder.embed_dim

        # Fusion layer to combine multi-view features
        total_views = len(input_dims)
        self.fusion_layer = nn.Linear(vit_embed_dim * total_views, embed_dim)

        # Projection layer to handle sequence features
        self.sequence_projection = nn.Linear(embed_dim, embed_dim)

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

    def _create_view_encoder(self, vit_model: str, patch_size: Tuple[int, int],
                           target_size: Tuple[int, int], freeze_backbone: bool,
                           max_sequence_length: int) -> nn.Module:
        """Create Vision Transformer encoder for a single radar view."""
        # Convert patch_size to tuple
        patch_size_tuple = (patch_size, patch_size)
        return ViewEncoder(vit_model, patch_size_tuple, target_size, freeze_backbone, max_sequence_length)

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

        # Handle different input formats
        if isinstance(radar_data, dict):
            # Multi-view format
            features = self._encode_multi_view(radar_data)
        elif isinstance(radar_data, torch.Tensor):
            # Single tensor format (assume it's already processed)
            features = radar_data
        else:
            raise ValueError(f"Unsupported radar data format: {type(radar_data)}")

        # Apply layer normalization and dropout
        features = self.layer_norm(features)
        features = self.dropout_layer(features)

        # Pool features to get final embedding
        pooled_features = features.mean(dim=1)

        # Apply sequence projection if needed
        if return_sequence and features.dim() == 3:
            sequence_features = self.sequence_projection(features)
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
                    "vit_model": self.vit_model
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
                    "vit_model": self.vit_model
                }
            )

    def _encode_multi_view(self, radar_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multi-view radar data using Vision Transformers.

        Args:
            radar_data: Dictionary with radar views

        Returns:
            Encoded features
        """
        view_features = []

        # Process each available view
        for view_name, encoder in self.view_encoders.items():
            if view_name in radar_data and radar_data[view_name] is not None:
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

                # Encode view using ViT
                encoded_view = encoder(view_data)  # [batch, num_patches, embed_dim]
                view_features.append(encoded_view)
            else:
                # Create dummy features if view is missing
                batch_size = self._get_batch_size(radar_data)
                dummy_features = torch.zeros(
                    batch_size, self.max_sequence_length, encoder.embed_dim,
                    device=next(self.parameters()).device
                )
                view_features.append(dummy_features)

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
        patch_size_range=config.get("patch_size_range", [32, 16]),
        patch_size_other=config.get("patch_size_other", [16, 16]),
        max_sequence_length=config.get("max_sequence_length", 196)
    )


# Component is automatically registered via @register_encoder decorator