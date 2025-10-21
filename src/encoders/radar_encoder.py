"""
Refactored radar encoder using the new abstraction layer.

This module provides a clean, extensible radar encoder implementation
that integrates with the mmExpert framework's core abstractions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..core.base import BaseEncoder, ModalityData, ModalityType, EncodingResult
from ..core.config import EncoderConfig
from ..core.registry import register_encoder
from ..core.factory import auto_factory


@register_encoder(
    name="radar_encoder",
    description="Radar encoder with multi-view processing",
    tags=["radar", "multimodal", "encoder"]
)
class RadarEncoder(BaseEncoder):
    """
    Enhanced radar encoder for processing mmWave radar data.

    This encoder supports:
    - Multi-view radar processing (range-time, doppler-time, azimuth-time)
    - Configurable architecture
    - Sequence-level encoding
    - Flexible input handling
    """

    def __init__(self,
                 embed_dim: int = 512,
                 input_dims: Dict[str, int] = None,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_sequence_length: int = 496,
                 use_layer_norm: bool = True,
                 activation: str = "gelu",
                 **kwargs):
        """
        Initialize radar encoder.

        Args:
            embed_dim: Embedding dimension
            input_dims: Input dimensions for each radar view
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_sequence_length: Maximum sequence length
            use_layer_norm: Whether to use layer normalization
            activation: Activation function
            **kwargs: Additional parameters
        """
        super().__init__(embed_dim=embed_dim, modality=ModalityType.RADAR, **kwargs)

        # Default input dimensions if not provided
        if input_dims is None:
            input_dims = {
                "range_time": 256,
                "doppler_time": 128,
                "azimuth_time": 128
            }

        self.input_dims = input_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.use_layer_norm = use_layer_norm

        # Create encoders for each radar view
        self.view_encoders = nn.ModuleDict()
        for view_name, input_dim in input_dims.items():
            self.view_encoders[view_name] = self._create_view_encoder(input_dim)

        # Fusion layer
        total_views = len(input_dims)
        self.fusion_layer = nn.Linear(embed_dim * total_views, embed_dim)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()

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

    def _create_view_encoder(self, input_dim: int) -> nn.Module:
        """Create encoder for a single radar view."""
        return nn.Sequential(
            nn.Conv1d(input_dim, self.embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.embed_dim // 2, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.max_sequence_length)
        )

    def _create_positional_encoding(self) -> nn.Embedding:
        """Create positional encoding for sequences."""
        return nn.Embedding(self.max_sequence_length, self.embed_dim)

    def _initialize_parameters(self) -> None:
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def encode(self,
               data: ModalityData,
               return_sequence: bool = False,
               **kwargs) -> EncodingResult:
        """
        Encode radar data.

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

        # Apply positional encoding if sequence
        if return_sequence and features.dim() == 3:
            seq_len = features.size(1)
            positions = torch.arange(seq_len, device=features.device)
            pos_encoding = self.positional_encoding(positions).unsqueeze(0)
            features = features + pos_encoding

        # Apply layer normalization and dropout
        features = self.layer_norm(features)
        features = self.dropout_layer(features)

        # Create encoding result
        if return_sequence and features.dim() == 3:
            # Return sequence features
            pooled_features = features.mean(dim=1)
            return EncodingResult(
                features=pooled_features,
                sequence_features=features,
                metadata={
                    "modality": "radar",
                    "sequence_length": features.size(1),
                    "embed_dim": self.embed_dim
                }
            )
        else:
            # Return pooled features
            return EncodingResult(
                features=features,
                metadata={
                    "modality": "radar",
                    "embed_dim": self.embed_dim
                }
            )

    def _encode_multi_view(self, radar_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multi-view radar data.

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

                # Ensure data is in the right format [batch, channels, time]
                if view_data.dim() == 3:
                    # Already in the right format
                    pass
                elif view_data.dim() == 2:
                    # Add batch dimension
                    view_data = view_data.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected tensor dimensions for {view_name}: {view_data.dim()}")

                # Encode view
                encoded_view = encoder(view_data)  # [batch, embed_dim, seq_len]
                encoded_view = encoded_view.transpose(1, 2)  # [batch, seq_len, embed_dim]
                view_features.append(encoded_view)
            else:
                # Create dummy features if view is missing
                batch_size = self._get_batch_size(radar_data)
                dummy_features = torch.zeros(
                    batch_size, self.max_sequence_length, self.embed_dim,
                    device=next(self.parameters()).device
                )
                view_features.append(dummy_features)

        # Concatenate view features
        if len(view_features) > 1:
            concatenated = torch.cat(view_features, dim=-1)  # [batch, seq_len, embed_dim * num_views]
            features = self.fusion_layer(concatenated)  # [batch, seq_len, embed_dim]
        else:
            features = view_features[0]

        return features

    def _get_batch_size(self, radar_data: Dict[str, torch.Tensor]) -> int:
        """Get batch size from radar data."""
        for view_data in radar_data.values():
            if view_data is not None:
                return view_data.size(0)
        return 1  # Default batch size

    def forward(self, range_data, doppler_data, azimuth_data, return_sequence=False):
        """
        Forward pass for backward compatibility.

        Args:
            range_data: Range-time data
            doppler_data: Doppler-time data
            azimuth_data: Azimuth-time data
            return_sequence: Whether to return sequence features

        Returns:
            Encoded features
        """
        # Create modality data
        radar_dict = {
            "range_time": range_data,
            "doppler_time": doppler_data,
            "azimuth_time": azimuth_data
        }

        # Remove None values
        radar_dict = {k: v for k, v in radar_dict.items() if v is not None}

        # Create ModalityData object
        modality_data = ModalityData(
            data=radar_dict,
            modality=ModalityType.RADAR,
            metadata={"format": "multi_view"}
        )

        # Encode
        result = self.encode(modality_data, return_sequence=return_sequence)

        if return_sequence:
            return result.sequence_features
        else:
            return result.features


# Factory function for creating radar encoder
def create_radar_encoder(config: EncoderConfig) -> RadarEncoder:
    """Create radar encoder from configuration."""
    return RadarEncoder(
        embed_dim=config.embed_dim,
        input_dims=config.get("input_dims"),
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_sequence_length=config.get("max_sequence_length", 496),
        use_layer_norm=config.get("use_layer_norm", True),
        activation=config.get("activation", "gelu")
    )


# Register factory
auto_factory._factory_map["encoder"]["radar_encoder"] = create_radar_encoder