"""Phi3 model with wave feature fusion support

This module uses the generic modeling_causal implementation to create
Phi3-specific model classes with wave feature fusion.
"""

from transformers import Phi3Model as _Phi3Model, Phi3ForCausalLM as _Phi3ForCausalLM
from .modeling_causal import create_wave_causal_model, create_wave_causal_lm

# Create wave-enabled Phi3 model classes using the generic factory
Phi3Model = create_wave_causal_model(_Phi3Model)
Phi3ForCausalLM = create_wave_causal_lm(_Phi3ForCausalLM, _Phi3Model)
