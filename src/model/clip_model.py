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

from ..core.base import BaseModel, ModalityData, ModalityType, EncodingResult
from ..core.config import ModelConfig
from ..core.registry import register_model
from ..core.factory import auto_factory
from ..core.injection import injectable, ServiceLifetime
from ..encoders.radar_encoder import RadarEncoder
from ..encoders.text_encoder import TextEncoder


@register_model(
    name="clip_model",
    description="CLIP model with abstraction layer integration",
    tags=["clip", "multimodal", "model"]
)
@injectable(ServiceLifetime.TRANSIENT)
class CLIPModel(BaseModel):
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
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            modality_types=[ModalityType(t) for t in (modality_types or ["radar", "text"])],
            **kwargs
        )

        self.embed_dim = embed_dim
        self.temperature = temperature
        self.use_siglip = use_siglip
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        # Initialize encoder configurations
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
                    "model_name": "bert-base-uncased",
                    "max_length": 77,
                    "pooling_strategy": "cls"
                }
            }

        # Create encoders
        self._create_encoders(encoder_configs)

        # Initialize similarity parameters
        if use_siglip:
            init_logit_scale = np.log(1 / temperature)
            self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
            self.logit_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.logit_scale = 1.0 / temperature

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
            similarity = similarity * self.logit_scale

        return similarity

    def forward(self,
                radar_data: Dict[str, torch.Tensor],
                text: List[str],
                return_sequences: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training/inference.

        Args:
            radar_data: Dictionary with radar data
            text: List of text captions
            return_sequences: Whether to return sequence features
            **kwargs: Additional arguments

        Returns:
            Tuple of (radar_features, text_features) or (radar_features, text_features, radar_seq, text_seq)
        """
        # Create modality data objects
        radar_modality_data = ModalityData(
            data=radar_data,
            modality=ModalityType.RADAR,
            metadata={"format": "multi_view"}
        )

        text_modality_data = ModalityData(
            data=text,
            modality=ModalityType.TEXT,
            metadata={"format": "string_list"}
        )

        # Encode data
        encoding_results = self.encode(
            {
                ModalityType.RADAR: radar_modality_data,
                ModalityType.TEXT: text_modality_data
            },
            return_sequences=return_sequences
        )

        # Extract features
        radar_features = encoding_results[ModalityType.RADAR].features
        text_features = encoding_results[ModalityType.TEXT].features

        # Normalize features
        radar_features = F.normalize(radar_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        if return_sequences:
            radar_seq = encoding_results[ModalityType.RADAR].sequence_features
            text_seq = encoding_results[ModalityType.TEXT].sequence_features

            # Normalize sequence features
            radar_seq = F.normalize(radar_seq, p=2, dim=-1)
            text_seq = F.normalize(text_seq, p=2, dim=-1)

            return radar_features, text_features, radar_seq, text_seq
        else:
            return radar_features, text_features

    def compute_loss(self,
                     radar_features: torch.Tensor,
                     text_features: torch.Tensor,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute CLIP loss.

        Args:
            radar_features: Encoded radar features
            text_features: Encoded text features
            **kwargs: Additional loss arguments

        Returns:
            Dictionary with loss values
        """
        batch_size = radar_features.size(0)

        # Compute similarity matrix
        logits = torch.matmul(radar_features, text_features.T) * self.logit_scale

        # Create labels
        labels = torch.arange(batch_size, device=radar_features.device)

        # Compute losses
        loss_radar_to_text = F.cross_entropy(logits, labels)
        loss_text_to_radar = F.cross_entropy(logits.T, labels)

        # Total loss
        total_loss = (loss_radar_to_text + loss_text_to_radar) / 2

        return {
            "loss_clip": total_loss,
            "loss_radar_to_text": loss_radar_to_text,
            "loss_text_to_radar": loss_text_to_radar
        }

    def get_training_parameters(self) -> List[torch.nn.Parameter]:
        """Get parameters for training."""
        params = []

        # Add encoder parameters
        if hasattr(self, 'radar_encoder'):
            params.extend(list(self.radar_encoder.parameters()))
        if hasattr(self, 'text_encoder'):
            params.extend(list(self.text_encoder.parameters()))

        # Add similarity parameters
        if self.use_siglip:
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

        if self.use_siglip:
            param_groups.append({
                'params': [self.logit_scale, self.logit_bias],
                'lr': self.learning_rate
            })

        optimizer = torch.optim.AdamW(param_groups, betas=(0.5, 0.9), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=0)

        return optimizer, scheduler

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


# Register factory
auto_factory._factory_map["model"]["clip_model"] = create_clip_model