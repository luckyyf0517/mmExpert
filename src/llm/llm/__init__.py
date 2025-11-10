# Import LLM model components
from .model_factory import ModelFactory
from .modeling_phi3 import Phi3Model, Phi3ForCausalLM
from .utils import process_wave_features

# Try to import Qwen2ForCausalLM if available
try:
    from .qwen3_model import Qwen2Model, Qwen2ForCausalLM
except ImportError:
    Qwen2Model = None
    Qwen2ForCausalLM = None

__all__ = [
    'process_wave_features',
    'Phi3Model',
    'Phi3ForCausalLM',
    'ModelFactory',
]

# Add Qwen2 classes to __all__ if available
if Qwen2Model is not None:
    __all__.extend(['Qwen2Model', 'Qwen2ForCausalLM'])

