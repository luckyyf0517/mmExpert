# Import LLM model components
from transformers import Phi3Model as _Phi3Model, Phi3ForCausalLM as _Phi3ForCausalLM
from transformers import Qwen2Model as _Qwen2Model, Qwen2ForCausalLM as _Qwen2ForCausalLM
from .modeling_causal import create_wave_causal_model, create_wave_causal_lm
from .utils import process_wave_features

# Create wave-enabled Phi3 model classes using the generic factory
Phi3Model = create_wave_causal_model(_Phi3Model)
Phi3ForCausalLM = create_wave_causal_lm(_Phi3ForCausalLM, _Phi3Model)

# Create wave-enabled Qwen2 model classes using the generic factory
Qwen2Model = create_wave_causal_model(_Qwen2Model)
Qwen2ForCausalLM = create_wave_causal_lm(_Qwen2ForCausalLM, _Qwen2Model)

# Import ModelFactory after model classes are created to avoid circular import
from .model_factory import ModelFactory

__all__ = [
    'process_wave_features',
    'Phi3Model',
    'Phi3ForCausalLM',
    'Qwen2Model',
    'Qwen2ForCausalLM',
    'ModelFactory',
]

