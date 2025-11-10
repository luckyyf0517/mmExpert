"""Base model classes for Vision2Seq models with mmwave feature support"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import PreTrainedModel
from typing import List, Optional, Union, Tuple
from transformers.modeling_outputs import ModelOutput
from termcolor import colored


class WaveModelBase(PreTrainedModel, ABC):
    """Abstract base class for Vision2Seq models with wave feature support"""

    @abstractmethod
    def _get_base_model_class(self):
        """Get the base model class"""
        pass

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_wave_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass for Vision2Seq models with wave feature support

        Args:
            pixel_values: Image pixel values [B, C, H, W]
            input_wave_embeds: Wave features [B, N_w, H] (already projected)
            input_ids: Text token ids [B, L]
            attention_mask: Attention mask [B, L]
            **kwargs: Additional arguments

        Returns:
            Model output (logits, hidden_states, etc.)
        """
        # Get the base class to call its forward method
        base_class = self._get_base_model_class()

        # Call base model forward with wave features as additional input
        return super(base_class, self).forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_wave_embeds=input_wave_embeds,  # Pass wave features directly
            **kwargs
        )


class WaveModelForVision2SeqBase(WaveModelBase, ABC):
    """Abstract base class for Vision2Seq models with wave feature support"""

    def __init__(self, config, **kwargs):
        # Extract dtype from kwargs for compatibility
        dtype = kwargs.pop('dtype', None)

        # Initialize the parent Vision2Seq model (this will be called by concrete subclasses)
        super().__init__(config, **kwargs)

        # Convert model to specified dtype if provided
        if dtype is not None:
            self.to(dtype=dtype)

        # mm_projection_layers will be initialized later via initialize_wave_projection()
        self.mm_projection_layers = None

    @abstractmethod
    def _get_base_model_class(self):
        """Get the base Vision2Seq model class"""
        pass

    def initialize_wave_projection(self, wave_feature_dim: int):
        """Initialize wave feature projection layer

        Args:
            wave_feature_dim: Dimension of wave features (radar encoder output)
        """
        if self.mm_projection_layers is not None:
            # Already initialized
            return

        # Create projection layer with model's dtype
        model_dtype = next(self.parameters()).dtype
        self.mm_projection_layers = nn.Linear(
            wave_feature_dim,
            self.config.text_config.hidden_size if hasattr(self.config, 'text_config') else self.config.hidden_size
        ).to(dtype=model_dtype)

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        input_wave_embeds=None,
        **kwargs,
    ):
        """Prepare inputs for generation, pass through input_wave_embeds"""
        # Get the base class to call its prepare_inputs_for_generation method
        base_class = self._get_base_model_class()
        model_inputs = super(base_class, self).prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs
        )

        # Pass through input_wave_embeds to forward
        model_inputs.update({
            "input_wave_embeds": input_wave_embeds,
        })

        return model_inputs

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_wave_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, ModelOutput]:
        """Forward for Vision2Seq models with wave feature support"""

        # Get the base class to call its forward method
        base_class = self._get_base_model_class()

        # Call base model forward with wave features only (no image input)
        # pixel_values is set to None since we don't use image input
        # input_wave_embeds are already projected and used as input_features for multimodal fusion
        outputs = super(base_class, self).forward(
            pixel_values=None,  # No image input
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_features=input_wave_embeds,  # Use wave features for multimodal fusion
            **kwargs
        )

        return outputs
