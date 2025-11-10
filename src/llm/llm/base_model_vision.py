"""Base model classes for Vision2Seq models with mmwave feature support"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import PreTrainedModel, AutoModelForVision2Seq
from typing import List, Optional, Union, Tuple
from transformers.modeling_outputs import ModelOutput
from termcolor import colored


class WaveModelForVision2SeqBase(PreTrainedModel, AutoModelForVision2Seq):
    """Abstract base class for Vision2Seq models with wave feature support

    This allows Vision2Seq models to accept pre-extracted wave features
    instead of processing images through the vision encoder.

    Usage:
        When input_wave_embeds is provided, it bypasses the vision encoder and
        directly uses the pre-extracted features (already projected to model dimension).
    """

    def __init__(self, config, **kwargs):
        """Initialize Vision2Seq model wrapper"""
        # Extract dtype from kwargs for compatibility
        dtype = kwargs.pop('dtype', None)

        # Initialize the parent Vision2Seq model (this will be called by concrete subclasses)
        super().__init__(config, **kwargs)

        # Convert model to specified dtype if provided
        if dtype is not None:
            self.to(dtype=dtype)

    @abstractmethod
    def _get_base_model_class(self):
        """Get the base Vision2Seq model class"""
        pass

    def forward(
        self,
        input_wave_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Forward pass with optional wave feature injection

        Args:
            input_wave_embeds: Pre-extracted wave features [B, N_v, D_v] (already projected)
                              If provided, bypasses vision encoder and uses these features directly
            pixel_values: Image pixel values [B, C, H, W]
                         Used only if input_wave_embeds is None
            input_ids: Text token ids [B, L]
            attention_mask: Attention mask [B, L]
            labels: Labels for loss computation [B, L]
            **kwargs: Additional arguments

        Returns:
            Model output (loss, logits, etc.)
        """
        # If input_wave_embeds provided, bypass vision encoder and use wave features directly
        if input_wave_embeds is not None:
            # Skip vision encoder processing and directly use wave features
            # We need to manually construct what the vision encoder would output

            # Get the base class to call its forward method
            base_class = self._get_base_model_class()

            # Create a modified kwargs that bypasses vision processing
            # For Vision2Seq models, we need to simulate the vision encoder output
            # input_wave_embeds should already be in the correct format for the model

            # Call the parent's forward but bypass vision encoder
            return super(base_class, self).forward(
                pixel_values=None,  # Don't use pixel_values when we have wave features
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                input_features=input_wave_embeds,  # Use wave features as vision features
                **kwargs
            )
        else:
            # Normal forward with pixel_values (standard Vision2Seq behavior)
            base_class = self._get_base_model_class()
            return super(base_class, self).forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

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
        if input_wave_embeds is not None:
            model_inputs.update({
                "input_wave_embeds": input_wave_embeds,
            })

        return model_inputs
