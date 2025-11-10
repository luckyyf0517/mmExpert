"""Qwen2 model with wave feature fusion support

This module uses the generic modeling_causal implementation to create
Qwen2-specific model classes with wave feature fusion.
"""

from transformers import Qwen2Model as _Qwen2Model, Qwen2ForCausalLM as _Qwen2ForCausalLM
from .modeling_causal import create_wave_causal_model, create_wave_causal_lm

# Create wave-enabled Qwen2 model classes using the generic factory
Qwen2Model = create_wave_causal_model(_Qwen2Model)
Qwen2ForCausalLM = create_wave_causal_lm(_Qwen2ForCausalLM, _Qwen2Model)

