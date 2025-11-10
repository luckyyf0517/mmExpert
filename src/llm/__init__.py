"""LLM module for mmExpert"""

from .trainer import WaveLLMTrainer
from .datamodule import WaveLLMDataModule
from .dataset import WaveCaptionDataset, DEFAULT_QUESTION_PROMPTS
from .llm.model_factory import ModelFactory
from .llm.modeling_phi3 import Phi3ForCausalLM
from .llm.utils import process_wave_features

# Try to import Qwen2ForCausalLM if available
try:
    from .llm.qwen3_model import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = None

__all__ = [
    'WaveLLMTrainer',
    'WaveLLMDataModule',
    'WaveCaptionDataset',
    'DEFAULT_QUESTION_PROMPTS',
    'ModelFactory',
    'process_wave_features',
    'Phi3ForCausalLM',
]

# Add Qwen2ForCausalLM to __all__ if available
if Qwen2ForCausalLM is not None:
    __all__.append('Qwen2ForCausalLM')