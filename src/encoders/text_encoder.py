"""
Refactored text encoder using the new abstraction layer.

This module provides a clean, extensible text encoder implementation
that integrates with the mmExpert framework's core abstractions.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, Optional, List, Union

from ..core.base import BaseEncoder, ModalityData, ModalityType, EncodingResult
from ..core.config import EncoderConfig
from ..core.registry import register_encoder
from ..core.factory import auto_factory


@register_encoder(
    name="text_encoder",
    description="Text encoder with transformer backbone",
    tags=["text", "transformer", "encoder"]
)
class TextEncoder(BaseEncoder):
    """
    Enhanced text encoder for processing text data.

    This encoder supports:
    - Multiple pre-trained transformer models
    - Configurable pooling strategies
    - Sequence-level encoding
    - Flexible tokenization
    """

    def __init__(self,
                 embed_dim: int = 512,
                 model_name: str = "bert-base-uncased",
                 max_length: int = 77,
                 pooling_strategy: str = "cls",
                 freeze_backbone: bool = False,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True,
                 **kwargs):
        """
        Initialize text encoder.

        Args:
            embed_dim: Output embedding dimension
            model_name: Pre-trained model name
            max_length: Maximum sequence length
            pooling_strategy: Pooling strategy ("cls", "mean", "max")
            freeze_backbone: Whether to freeze backbone parameters
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            **kwargs: Additional parameters
        """
        super().__init__(embed_dim=embed_dim, modality=ModalityType.TEXT, **kwargs)

        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.freeze_backbone = freeze_backbone
        self.use_layer_norm = use_layer_norm

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)

            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

        # Get backbone embedding dimension
        backbone_dim = self.backbone.config.hidden_size

        # Projection layer to match embed_dim
        self.projection = nn.Linear(backbone_dim, embed_dim)

        # Optional dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Sequence support
        self._supports_sequence = True

    def encode(self,
               data: ModalityData,
               return_sequence: bool = False,
               **kwargs) -> EncodingResult:
        """
        Encode text data.

        Args:
            data: Text modality data
            return_sequence: Whether to return sequence-level features
            **kwargs: Additional encoding arguments

        Returns:
            Encoding result with features and metadata
        """
        # Extract text
        text_data = data.data

        # Handle different input formats
        if isinstance(text_data, (list, tuple)):
            # List of strings
            text_list = list(text_data)
        elif isinstance(text_data, str):
            # Single string
            text_list = [text_data]
        elif isinstance(text_data, torch.Tensor):
            # Already tokenized
            return self._encode_pre_tokenized(text_data, return_sequence)
        else:
            raise ValueError(f"Unsupported text data format: {type(text_data)}")

        # Tokenize text
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode with backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.backbone(**inputs)

        # Extract hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, backbone_dim]

        # Apply pooling
        if return_sequence:
            # Return full sequence after projection
            pooled_features = self._apply_pooling(hidden_states, inputs.attention_mask)
            sequence_features = self.projection(hidden_states)
            sequence_features = self.layer_norm(sequence_features)
            sequence_features = self.dropout_layer(sequence_features)
        else:
            # Return pooled features
            pooled_features = self._apply_pooling(hidden_states, inputs.attention_mask)
            sequence_features = None

        # Project to embed_dim
        pooled_features = self.projection(pooled_features)
        pooled_features = self.layer_norm(pooled_features)
        pooled_features = self.dropout_layer(pooled_features)

        # Create encoding result
        return EncodingResult(
            features=pooled_features,
            sequence_features=sequence_features,
            attention_mask=inputs.attention_mask,
            metadata={
                "modality": "text",
                "model_name": self.model_name,
                "pooling_strategy": self.pooling_strategy,
                "sequence_length": hidden_states.size(1),
                "embed_dim": self.embed_dim
            }
        )

    def _encode_pre_tokenized(self,
                             tokenized_data: torch.Tensor,
                             return_sequence: bool = False) -> EncodingResult:
        """
        Encode already tokenized data.

        Args:
            tokenized_data: Pre-tokenized input IDs
            return_sequence: Whether to return sequence features

        Returns:
            Encoding result
        """
        # Create attention mask (assuming padding token is 0)
        attention_mask = (tokenized_data != 0).long()

        # Encode with backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.backbone(input_ids=tokenized_data, attention_mask=attention_mask)

        # Extract hidden states
        hidden_states = outputs.last_hidden_state

        # Apply pooling
        if return_sequence:
            pooled_features = self._apply_pooling(hidden_states, attention_mask)
            sequence_features = self.projection(hidden_states)
            sequence_features = self.layer_norm(sequence_features)
            sequence_features = self.dropout_layer(sequence_features)
        else:
            pooled_features = self._apply_pooling(hidden_states, attention_mask)
            sequence_features = None

        # Project to embed_dim
        pooled_features = self.projection(pooled_features)
        pooled_features = self.layer_norm(pooled_features)
        pooled_features = self.dropout_layer(pooled_features)

        return EncodingResult(
            features=pooled_features,
            sequence_features=sequence_features,
            attention_mask=attention_mask,
            metadata={
                "modality": "text",
                "model_name": self.model_name,
                "pooling_strategy": self.pooling_strategy,
                "pre_tokenized": True
            }
        )

    def _apply_pooling(self,
                      hidden_states: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling strategy to hidden states.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled features [batch, hidden_dim]
        """
        if self.pooling_strategy == "cls":
            # Use CLS token (first token)
            return hidden_states[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            # Max pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            masked_hidden = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            return torch.max(masked_hidden, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def tokenize(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text inputs.

        Args:
            text: Text or list of texts

        Returns:
            Dictionary with tokenized inputs
        """
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        return inputs

    def forward(self, text, device='cuda', return_sequence=False):
        """
        Forward pass for backward compatibility.

        Args:
            text: Text input
            device: Device to run on
            return_sequence: Whether to return sequence features

        Returns:
            Encoded features
        """
        # Create modality data
        modality_data = ModalityData(
            data=text,
            modality=ModalityType.TEXT,
            metadata={"device": device}
        )

        # Encode
        result = self.encode(modality_data, return_sequence=return_sequence)

        if return_sequence:
            return result.sequence_features
        else:
            return result.features


# Factory function for creating text encoder
def create_text_encoder(config: EncoderConfig) -> TextEncoder:
    """Create text encoder from configuration."""
    return TextEncoder(
        embed_dim=config.embed_dim,
        model_name=config.get("model_name", "bert-base-uncased"),
        max_length=config.get("max_length", 77),
        pooling_strategy=config.get("pooling_strategy", "cls"),
        freeze_backbone=config.get("freeze_backbone", False),
        dropout=config.dropout,
        use_layer_norm=config.get("use_layer_norm", True)
    )


# Register factory
auto_factory._factory_map["encoder"]["text_encoder"] = create_text_encoder