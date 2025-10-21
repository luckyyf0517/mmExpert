import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import torchvision
from typing import Tuple
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import _create_vision_transformer
from termcolor import colored


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

        # Validate input dimensions
        batch_size = None
        for name, data in [("range", range_data), ("doppler", doppler_data), ("azimuth", azimuth_data)]:
            if data is not None:
                if data.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for {name} data, got {data.dim()}D tensor with shape {data.shape}")

                if batch_size is None:
                    batch_size = data.size(0)
                elif batch_size != data.size(0):
                    raise ValueError(f"Inconsistent batch sizes: {name} has batch size {data.size(0)}, expected {batch_size}")

        # Encode each view based on configuration and data availability
        if self.use_range and range_data is not None:
            self._validate_radar_view_data(range_data, "range", expected_height=256)
            range_data_2d = range_data.unsqueeze(1)    # [b, 1, 256, T]
            range_features = self.encode_view(self.range_encoder, range_data_2d)  # [b, n_range, embed_dim]
            self._validate_encoded_features(range_features, "range", batch_size)
            features_list.append(range_features)

        if self.use_doppler and doppler_data is not None:
            self._validate_radar_view_data(doppler_data, "doppler", expected_height=128)
            doppler_data_2d = doppler_data.unsqueeze(1)  # [b, 1, 128, T]
            doppler_features = self.encode_view(self.doppler_encoder, doppler_data_2d)  # [b, n_doppler, embed_dim]
            self._validate_encoded_features(doppler_features, "doppler", batch_size)
            features_list.append(doppler_features)

        if self.use_azimuth and azimuth_data is not None:
            self._validate_radar_view_data(azimuth_data, "azimuth", expected_height=128)
            azimuth_data_2d = azimuth_data.unsqueeze(1)  # [b, 1, 128, T]
            azimuth_features = self.encode_view(self.azimuth_encoder, azimuth_data_2d)  # [b, n_azimuth, embed_dim]
            self._validate_encoded_features(azimuth_features, "azimuth", batch_size)
            features_list.append(azimuth_features)

        # Check if we have any features
        if len(features_list) == 0:
            raise ValueError("No valid radar data provided (all inputs are None)")

        # Fuse features
        fused_features = self._fuse_features(features_list)

        return fused_features

    def _validate_radar_view_data(self, data: torch.Tensor, view_name: str, expected_height: int):
        """Validate radar view data dimensions and properties."""
        if data is None:
            return

        # Check dimension
        if data.dim() != 3:
            raise ValueError(f"{view_name} data should be 3D [batch, height, time], got {data.dim()}D: {data.shape}")

        batch_size, height, time_frames = data.shape

        # Check expected dimensions
        if height != expected_height:
            print(f"[WARNING] {view_name} data has height {height}, expected {expected_height}")

        if batch_size <= 0:
            raise ValueError(f"Invalid batch size {batch_size} for {view_name} data")

        if time_frames <= 0:
            raise ValueError(f"Invalid time frames {time_frames} for {view_name} data")

        # Check for NaN or infinite values
        if torch.isnan(data).any():
            raise ValueError(f"{view_name} data contains NaN values")

        if torch.isinf(data).any():
            raise ValueError(f"{view_name} data contains infinite values")

    def _validate_encoded_features(self, features: torch.Tensor, view_name: str, expected_batch_size: int):
        """Validate encoded feature dimensions."""
        if features is None:
            return

        # Check dimension
        if features.dim() != 3:
            raise ValueError(f"Encoded {view_name} features should be 3D [batch, seq_len, embed_dim], got {features.dim()}D: {features.shape}")

        batch_size, seq_len, embed_dim = features.shape

        # Check batch size
        if batch_size != expected_batch_size:
            raise ValueError(f"Encoded {view_name} features batch size {batch_size} doesn't match input {expected_batch_size}")

        # Check feature dimensions
        if embed_dim != self.embed_dim:
            raise ValueError(f"Encoded {view_name} features embed_dim {embed_dim} doesn't match expected {self.embed_dim}")

        # Check for NaN or infinite values
        if torch.isnan(features).any():
            raise ValueError(f"Encoded {view_name} features contain NaN values")

        if torch.isinf(features).any():
            raise ValueError(f"Encoded {view_name} features contain infinite values")

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