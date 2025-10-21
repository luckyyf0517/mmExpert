"""
CLIP model using the new abstraction layer.

This module provides a clean, extensible CLIP implementation
that integrates with the mmExpert framework's core abstractions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

import pytorch_lightning as pl
from torchmetrics import Metric

from ..core.base import BaseModel, ModalityData, ModalityType, EncodingResult
from ..core.config import ModelConfig
from ..core.registry import register_model
from ..core.factory import auto_factory
from ..core.injection import injectable, ServiceLifetime
from ..encoders.radar_encoder import RadarEncoder
from ..encoders.text_encoder import TextEncoder
from ..model.clip_loss import ClipLoss, SigLipLoss, create_loss
from ..model.sequence_similarity import SequenceSimilarity


@register_model(
    name="clip_model",
    description="CLIP model with abstraction layer integration",
    tags=["clip", "multimodal", "model"]
)
@injectable(ServiceLifetime.TRANSIENT)
class CLIPModel(pl.LightningModule):
    """
    CLIP model using the new abstraction layer.

    This model provides:
    - Clean separation of concerns
    - Flexible encoder configuration
    - Improved similarity computation
    - Better error handling
    - Extensible architecture
    """

    def __init__(self,
                 name: str = "clip_model",
                 modality_types: List[str] = None,
                 embed_dim: int = 512,
                 temperature: float = 0.07,
                 use_siglip: bool = False,
                 learning_rate: float = 1e-4,
                 max_epochs: int = 50,
                 encoder_configs: Dict[str, Any] = None,
                 # Distributed training parameters
                 local_loss: bool = False,
                 gather_with_grad: bool = False,
                 cache_labels: bool = True,
                 rank: int = 0,
                 world_size: int = 1,
                 use_horovod: bool = False,
                 loss_dist_impl: Optional[str] = None,
                 # Sequence similarity parameters
                 use_sequence_similarity: bool = False,
                 sequence_similarity_type: str = "combined",
                 sequence_similarity_weight: float = 0.5,
                 sequence_similarity_window_size: int = 16,
                 **kwargs):
        """
        Initialize CLIP model.

        Args:
            name: Model name
            modality_types: List of supported modality types
            embed_dim: Embedding dimension
            temperature: Temperature parameter for similarity
            use_siglip: Whether to use SigLIP loss
            learning_rate: Learning rate for training
            max_epochs: Maximum number of training epochs
            encoder_configs: Configuration for encoders
            local_loss: Whether to compute loss locally (distributed training)
            gather_with_grad: Whether to gather features with gradients (distributed training)
            cache_labels: Whether to cache labels for efficiency
            rank: Rank of current process (distributed training)
            world_size: Total number of processes (distributed training)
            use_horovod: Whether to use Horovod for distributed training
            loss_dist_impl: Distributed implementation for SigLIP loss
            use_sequence_similarity: Whether to enable sequence similarity computation
            sequence_similarity_type: Type of sequence similarity ("global", "local", "attention", "temporal", "combined")
            sequence_similarity_weight: Weight for sequence similarity loss
            sequence_similarity_window_size: Window size for local similarity
            **kwargs: Additional parameters
        """
        super().__init__()

        # Store model properties
        self.name = name
        self.modality_types = [ModalityType(t) for t in (modality_types or ["radar", "text"])]

        self.embed_dim = embed_dim
        self.temperature = temperature
        self.use_siglip = use_siglip
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        # Distributed training parameters
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.loss_dist_impl = loss_dist_impl

        # Sequence similarity parameters
        self.use_sequence_similarity = use_sequence_similarity
        self.sequence_similarity_type = sequence_similarity_type
        self.sequence_similarity_weight = sequence_similarity_weight
        self.sequence_similarity_window_size = sequence_similarity_window_size

        # Lightning specific - only save parameters that are actually used
        self.save_hyperparameters(
            'name', 'modality_types', 'embed_dim', 'temperature', 
            'use_siglip', 'learning_rate', 'max_epochs', 'encoder_configs'
        )

        # Use default configuration if none provided
        if encoder_configs is None:
            encoder_configs = {
                "radar": {
                    "embed_dim": embed_dim,
                    "num_layers": 4,
                    "num_heads": 8,
                    "dropout": 0.1
                },
                "text": {
                    "embed_dim": embed_dim,
                    "model_name": "sentence-transformers/paraphrase-MiniLM-L6-v2",
                    "max_length": 77,
                    "pooling_strategy": "cls"
                }
            }

        # Initialize encoders storage for compatibility
        self._encoders = {}
        self._processors = {}

        # Create encoders
        self._create_encoders(encoder_configs)

        # Create loss function using the existing clip_loss implementations
        self.criterion = self._create_loss_function()

        # Initialize similarity parameters (only if not using SigLip)
        if not use_siglip:
            self.logit_scale = 1.0 / temperature

        # Create sequence similarity module if needed
        if self.use_sequence_similarity:
            self.sequence_similarity = SequenceSimilarity(
                embed_dim=self.embed_dim,
                similarity_type=self.sequence_similarity_type,
                window_size=self.sequence_similarity_window_size,
                temperature=self.temperature
            )

        # Training parameters
        self._training_step = 0

    def _create_encoders(self, encoder_configs: Dict[str, Any]) -> None:
        """Create encoders based on configurations."""
        # Create radar encoder
        if "radar" in encoder_configs:
            radar_config = encoder_configs["radar"]
            self.radar_encoder = RadarEncoder(**radar_config)
            self.add_encoder(ModalityType.RADAR, self.radar_encoder)

        # Create text encoder
        if "text" in encoder_configs:
            text_config = encoder_configs["text"]
            self.text_encoder = TextEncoder(**text_config)
            self.add_encoder(ModalityType.TEXT, self.text_encoder)

    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        if self.use_siglip:
            return SigLipLoss(
                cache_labels=self.cache_labels,
                rank=self.rank,
                world_size=self.world_size,
                dist_impl=self.loss_dist_impl,
            )
        else:
            return ClipLoss(
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                cache_labels=self.cache_labels,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

    def _encode_data(self,
                     radar_data: Dict[str, torch.Tensor],
                     text_data: List[str],
                     return_sequences: bool = False,
                     **kwargs) -> Dict[ModalityType, EncodingResult]:
        """
        Unified encoding logic that converts raw data to feature representations.

        Args:
            radar_data: Dictionary of radar data
            text_data: List of text data
            return_sequences: Whether to return sequence features
            **kwargs: Additional parameters

        Returns:
            Dictionary of encoding results
        """
        # Create modality data objects
        modality_data = {}
        if radar_data is not None:
            modality_data[ModalityType.RADAR] = ModalityData(
                data=radar_data,
                modality=ModalityType.RADAR,
                metadata={"format": "multi_view"}
            )
        if text_data is not None:
            modality_data[ModalityType.TEXT] = ModalityData(
                data=text_data,
                modality=ModalityType.TEXT,
                metadata={"format": "string_list"}
            )

        # Determine whether to use sequence features based on sequence similarity configuration
        use_sequences = return_sequences or self.use_sequence_similarity

        # Encode data
        encoding_results = self.encode(modality_data, return_sequences=use_sequences, **kwargs)

        return encoding_results

    def encode(self,
               data: Dict[ModalityType, ModalityData],
               return_sequences: bool = False,
               **kwargs) -> Dict[ModalityType, EncodingResult]:
        """
        Encode multimodal data.

        Args:
            data: Dictionary mapping modalities to data
            return_sequences: Whether to return sequence-level features
            **kwargs: Additional encoding arguments

        Returns:
            Dictionary of encoding results for each modality
        """
        results = {}

        for modality, modality_data in data.items():
            encoder = self.get_encoder(modality)
            if encoder is None:
                raise ValueError(f"No encoder found for modality: {modality}")

            # Encode data
            result = encoder.encode(modality_data, return_sequence=return_sequences, **kwargs)
            results[modality] = result

        return results

    def compute_similarity(self,
                          features_1: Dict[ModalityType, torch.Tensor],
                          features_2: Dict[ModalityType, torch.Tensor],
                          **kwargs) -> torch.Tensor:
        """
        Compute similarity between feature sets.

        Args:
            features_1: First set of features
            features_2: Second set of features
            **kwargs: Additional similarity arguments

        Returns:
            Similarity tensor
        """
        # For CLIP, we typically compute similarity between radar and text features
        radar_features_1 = features_1.get(ModalityType.RADAR)
        text_features_1 = features_1.get(ModalityType.TEXT)
        radar_features_2 = features_2.get(ModalityType.RADAR)
        text_features_2 = features_2.get(ModalityType.TEXT)

        # Normalize features
        if radar_features_1 is not None:
            radar_features_1 = F.normalize(radar_features_1, p=2, dim=-1)
        if text_features_1 is not None:
            text_features_1 = F.normalize(text_features_1, p=2, dim=-1)
        if radar_features_2 is not None:
            radar_features_2 = F.normalize(radar_features_2, p=2, dim=-1)
        if text_features_2 is not None:
            text_features_2 = F.normalize(text_features_2, p=2, dim=-1)

        # Compute similarity matrix
        if radar_features_1 is not None and text_features_2 is not None:
            # Radar to text similarity
            similarity = torch.matmul(radar_features_1, text_features_2.T)
        elif text_features_1 is not None and radar_features_2 is not None:
            # Text to radar similarity
            similarity = torch.matmul(text_features_1, radar_features_2.T)
        elif radar_features_1 is not None and radar_features_2 is not None:
            # Radar to radar similarity
            similarity = torch.matmul(radar_features_1, radar_features_2.T)
        elif text_features_1 is not None and text_features_2 is not None:
            # Text to text similarity
            similarity = torch.matmul(text_features_1, text_features_2.T)
        else:
            raise ValueError("No valid feature pairs found for similarity computation")

        # Apply temperature scaling
        if not self.use_siglip:
            if isinstance(self.logit_scale, torch.nn.Parameter):
                similarity = similarity * torch.exp(self.logit_scale) if self.logit_scale.item() > 0 else similarity * self.logit_scale
            else:
                similarity = similarity * self.logit_scale

        return similarity

    def forward(self,
                radar_data: Dict[str, torch.Tensor],
                text: List[str],
                return_sequences: bool = False,
                compute_loss: bool = False,
                **kwargs):
        """
        Unified forward pass entry point.

        Args:
            radar_data: Dictionary of radar data
            text: List of text data
            return_sequences: Whether to return sequence features
            compute_loss: Whether to compute loss (for training)
            **kwargs: Additional parameters

        Returns:
            If compute_loss=True: Returns loss dictionary
            If return_sequences=True: Returns (radar_features, text_features, radar_seq, text_seq)
            Otherwise: Returns (radar_features, text_features)
        """
        # Unified encoding
        encoding_results = self._encode_data(radar_data, text, return_sequences, **kwargs)

        # Extract features
        radar_features = encoding_results[ModalityType.RADAR].features
        text_features = encoding_results[ModalityType.TEXT].features

        # Normalize features
        radar_features = F.normalize(radar_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Return different results based on requirements
        if compute_loss:
            return self._compute_losses_from_features(radar_features, text_features, encoding_results)
        elif return_sequences:
            radar_seq = encoding_results[ModalityType.RADAR].sequence_features
            text_seq = encoding_results[ModalityType.TEXT].sequence_features

            # Normalize sequence features
            if radar_seq is not None:
                radar_seq = F.normalize(radar_seq, p=2, dim=-1)
            if text_seq is not None:
                text_seq = F.normalize(text_seq, p=2, dim=-1)

            return radar_features, text_features, radar_seq, text_seq
        else:
            return radar_features, text_features

    def compute_loss(self,
                     radar_features: torch.Tensor,
                     text_features: torch.Tensor,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Pure loss computation with pre-encoded features as input.

        Args:
            radar_features: Pre-encoded radar features [B, D]
            text_features: Pre-encoded text features [B, D]
            **kwargs: Additional parameters

        Returns:
            Dictionary of loss values
        """
        return self._compute_clip_loss_only(radar_features, text_features)

    def _compute_losses_from_features(self,
                                    radar_features: torch.Tensor,
                                    text_features: torch.Tensor,
                                    encoding_results: Dict[ModalityType, EncodingResult]) -> Dict[str, torch.Tensor]:
        """
        Compute all losses from features, including standard CLIP loss and sequence similarity loss.

        Args:
            radar_features: Radar features
            text_features: Text features
            encoding_results: Encoding results (containing sequence features)

        Returns:
            Dictionary of loss values
        """
        losses = self._compute_clip_loss_only(radar_features, text_features)

        # If sequence similarity is enabled, compute sequence loss
        if self.use_sequence_similarity:
            radar_seq = encoding_results[ModalityType.RADAR].sequence_features
            text_seq = encoding_results[ModalityType.TEXT].sequence_features

            if radar_seq is not None and text_seq is not None:
                # Compute sequence similarity matrix
                seq_similarities = self.sequence_similarity(radar_seq, text_seq)

                # Convert similarity matrix to loss
                batch_size = radar_features.size(0)
                seq_labels = torch.arange(batch_size, device=radar_features.device)
                seq_loss = F.cross_entropy(seq_similarities, seq_labels)

                losses["loss_seq"] = seq_loss * self.sequence_similarity_weight

                # Total loss
                losses["loss_total"] = losses["loss_clip"] + losses["loss_seq"]

        return losses

    def _compute_clip_loss_only(self,
                               radar_features: torch.Tensor,
                               text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute standard CLIP loss only.

        Args:
            radar_features: Radar features (should be 2D and normalized)
            text_features: Text features (should be 2D and normalized)

        Returns:
            CLIP loss dictionary
        """
        # Use integrated loss function
        if self.use_siglip:
            # SigLIP loss
            if hasattr(self, 'logit_scale') and hasattr(self, 'logit_bias'):
                total_loss = self.criterion(radar_features, text_features, self.logit_scale, self.logit_bias)
            else:
                default_logit_scale = np.log(1 / self.temperature)
                default_logit_bias = torch.tensor(0.0, device=radar_features.device)
                total_loss = self.criterion(radar_features, text_features, default_logit_scale, default_logit_bias)

            return {"loss_clip": total_loss}
        else:
            # Standard CLIP loss
            total_loss = self.criterion(radar_features, text_features, self.logit_scale)

            # For compatibility with existing logging, compute individual losses
            batch_size = radar_features.size(0)
            logits_per_image, logits_per_text = self.criterion.get_logits(radar_features, text_features, self.logit_scale)
            labels = self.criterion.get_ground_truth(radar_features.device, logits_per_image.shape[0])

            loss_radar_to_text = F.cross_entropy(logits_per_image, labels)
            loss_text_to_radar = F.cross_entropy(logits_per_text, labels)

            return {
                "loss_clip": total_loss,
                "loss_radar_to_text": loss_radar_to_text,
                "loss_text_to_radar": loss_text_to_radar
            }

    def compute_clip_loss(self, modality_data: Dict[ModalityType, ModalityData]) -> Dict[str, torch.Tensor]:
        """
        Compute CLIP loss from modality data.

        Args:
            modality_data: Dictionary mapping modalities to modality data

        Returns:
            Dictionary with loss values
        
        Raises:
            ValueError: If required modalities are missing or feature shapes are invalid
        """
        # Check that both modalities are present
        if ModalityType.TEXT not in modality_data:
            raise ValueError(
                f"TEXT modality missing from input data. "
                f"Available modalities: {list(modality_data.keys())}. "
                f"This indicates a data loading issue."
            )
        
        if ModalityType.RADAR not in modality_data:
            raise ValueError(
                f"RADAR modality missing from input data. "
                f"Available modalities: {list(modality_data.keys())}. "
                f"This indicates a data loading issue."
            )

        # Encode data
        encoding_results = self.encode(modality_data, return_sequences=False)

        # Extract features
        radar_features = encoding_results[ModalityType.RADAR].features
        text_features = encoding_results[ModalityType.TEXT].features
        
        # Validate feature shapes - must be 2D [batch_size, embed_dim]
        if len(radar_features.shape) != 2:
            raise ValueError(
                f"Expected radar_features to be 2D [batch_size, embed_dim], "
                f"got shape {radar_features.shape}. "
                f"This indicates an encoder configuration issue. "
                f"Make sure return_sequences=False is properly handled in RadarEncoder."
            )
        
        if len(text_features.shape) != 2:
            raise ValueError(
                f"Expected text_features to be 2D [batch_size, embed_dim], "
                f"got shape {text_features.shape}. "
                f"This indicates an encoder configuration issue. "
                f"Make sure return_sequences=False is properly handled in TextEncoder."
            )

        # Use new method to compute losses
        return self._compute_losses_from_features(radar_features, text_features, encoding_results)

    def training_step(self, batch, batch_idx):
        """Lightning training step - using unified forward method."""
        # Extract data from batch
        radar_data = batch.get('radar', batch.get('radar_data'))
        text_data = batch.get('text', batch.get('caption'))

        # Get batch size for logging
        batch_size = self._get_batch_size_from_batch(batch)

        # Data validation
        if radar_data is None:
            self.print("Warning: No radar data found in batch, using dummy data")
            radar_data = {"range_time": torch.zeros(batch_size, 256, 100),
                         "doppler_time": torch.zeros(batch_size, 128, 100),
                         "azimuth_time": torch.zeros(batch_size, 128, 100)}
        if text_data is None:
            self.print("Warning: No text data found in batch, using dummy data")
            text_data = ["dummy text"] * batch_size

        # ✅ Use unified forward method to compute loss
        losses = self.forward(radar_data, text_data, compute_loss=True)

        # Get main loss
        main_loss = losses.get('loss_total', losses['loss_clip'])

        # Log losses
        self._log_training_losses(losses, batch_size, 'train')

        return main_loss

    def _get_batch_size_from_batch(self, batch) -> int:
        """Get batch size from batch"""
        radar_data = batch.get('radar', batch.get('radar_data'))
        text_data = batch.get('text', batch.get('caption'))

        if radar_data is not None:
            if isinstance(radar_data, dict):
                batch_size = next(iter(radar_data.values())).size(0)
            else:
                batch_size = radar_data.size(0)
        elif text_data is not None:
            batch_size = len(text_data) if isinstance(text_data, list) else text_data.size(0)
        else:
            batch_size = 1
        return batch_size

    def _log_training_losses(self, losses: Dict[str, torch.Tensor], batch_size: int, prefix: str):
        """Log training losses"""
        # Standard CLIP losses
        if 'loss_clip' in losses:
            self.log(f'{prefix}/loss_clip', losses['loss_clip'], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if 'loss_radar_to_text' in losses:
            self.log(f'{prefix}/loss_radar_to_text', losses['loss_radar_to_text'], on_step=True, on_epoch=True, batch_size=batch_size)
        if 'loss_text_to_radar' in losses:
            self.log(f'{prefix}/loss_text_to_radar', losses['loss_text_to_radar'], on_step=True, on_epoch=True, batch_size=batch_size)

        # Sequence similarity loss
        if 'loss_seq' in losses:
            self.log(f'{prefix}/loss_seq', losses['loss_seq'], on_step=True, on_epoch=True, batch_size=batch_size)

        # Total loss
        if 'loss_total' in losses:
            self.log(f'{prefix}/loss_total', losses['loss_total'], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def validation_step(self, batch, batch_idx):
        """Lightning validation step - using unified forward method."""
        # Extract data from batch
        radar_data = batch.get('radar', batch.get('radar_data'))
        text_data = batch.get('text', batch.get('caption'))

        # Get batch size for logging
        batch_size = self._get_batch_size_from_batch(batch)

        # Data validation
        if radar_data is None:
            self.print("Warning: No radar data found in validation batch, using dummy data")
            radar_data = {"range_time": torch.zeros(batch_size, 256, 100),
                         "doppler_time": torch.zeros(batch_size, 128, 100),
                         "azimuth_time": torch.zeros(batch_size, 128, 100)}
        if text_data is None:
            self.print("Warning: No text data found in validation batch, using dummy data")
            text_data = ["dummy text"] * batch_size

        # ✅ Use unified forward method to compute loss
        losses = self.forward(radar_data, text_data, compute_loss=True)

        # Get main loss
        main_loss = losses.get('loss_total', losses['loss_clip'])

        # Log losses
        self._log_training_losses(losses, batch_size, 'valid')

        return main_loss

    def test_step(self, batch, batch_idx):
        """Lightning test step."""
        return self.validation_step(batch, batch_idx)

    def add_encoder(self, modality: ModalityType, encoder):
        """Add an encoder for a specific modality."""
        self._encoders[modality] = encoder

    def get_encoder(self, modality: ModalityType):
        """Get encoder for a specific modality."""
        return self._encoders.get(modality)

    def get_training_parameters(self) -> List[torch.nn.Parameter]:
        """Get parameters for training."""
        params = []

        # Add encoder parameters
        if hasattr(self, 'radar_encoder'):
            params.extend(list(self.radar_encoder.parameters()))
        if hasattr(self, 'text_encoder'):
            params.extend(list(self.text_encoder.parameters()))

        # Add loss function parameters
        if hasattr(self, 'criterion'):
            params.extend(list(self.criterion.parameters()))

        # Add similarity parameters
        if self.use_siglip and hasattr(self, 'logit_scale') and hasattr(self, 'logit_bias'):
            params.extend([self.logit_scale, self.logit_bias])

        return params

    def configure_optimizers(self):
        """Configure optimizers for training."""
        # Create parameter groups with different learning rates
        param_groups = []

        if hasattr(self, 'text_encoder'):
            param_groups.append({
                'params': self.text_encoder.parameters(),
                'lr': self.learning_rate / 2
            })

        if hasattr(self, 'radar_encoder'):
            param_groups.append({
                'params': self.radar_encoder.parameters(),
                'lr': self.learning_rate
            })

        if hasattr(self, 'criterion'):
            param_groups.append({
                'params': self.criterion.parameters(),
                'lr': self.learning_rate
            })

        if self.use_siglip and hasattr(self, 'logit_scale') and hasattr(self, 'logit_bias'):
            param_groups.append({
                'params': [self.logit_scale, self.logit_bias],
                'lr': self.learning_rate
            })

        optimizer = torch.optim.AdamW(param_groups, betas=(0.5, 0.9), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=0)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def save_pretrained(self, save_directory: str) -> None:
        """Save model configuration and weights."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save configuration
        config = {
            "name": self.name,
            "embed_dim": self.embed_dim,
            "temperature": self.temperature,
            "use_siglip": self.use_siglip,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "modality_types": [m.value for m in self.modality_types]
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, model_directory: str) -> 'CLIPModel':
        """Load model from saved directory."""
        import os
        import json

        # Load configuration
        with open(os.path.join(model_directory, "config.json"), "r") as f:
            config = json.load(f)

        # Create model
        model = cls(**config)

        # Load weights
        model.load_state_dict(torch.load(os.path.join(model_directory, "pytorch_model.bin")))

        return model


# Factory function for creating CLIP model
def create_clip_model(config: ModelConfig) -> CLIPModel:
    """Create CLIP model from configuration."""
    return CLIPModel(
        name=config.name,
        modality_types=config.modality_types,
        embed_dim=config.embed_dim,
        temperature=config.temperature,
        use_siglip=config.get("use_siglip", False),
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        encoder_configs=config.get("encoder_configs")
    )


# Component is automatically registered via @register_model decorator