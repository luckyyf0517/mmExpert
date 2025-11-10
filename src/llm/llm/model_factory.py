"""Factory class for creating models and tokenizers with concrete model implementations"""

import torch
import torch.nn as nn
from typing import Dict, Type, Any
from transformers import (AutoConfig, AutoTokenizer,
                         Phi3Model as _Phi3Model, Phi3ForCausalLM as _Phi3ForCausalLM,
                         Qwen2Model as _Qwen2Model, Qwen2ForCausalLM as _Qwen2ForCausalLM)
from .base_model_casual import WaveModelBase, WaveModelForCausalBase


def create_wave_model_class(base_model_class: Type, base_model_name: str) -> Type:
    """
    Factory function to create a wave-enhanced model class

    Args:
        base_model_class: The base model class (e.g., _Phi3Model, _Qwen2Model)
        base_model_name: Name for the created class (e.g., 'Phi3Model')

    Returns:
        A new class that inherits from WaveModelBase and the base model class
    """
    class WaveModel(WaveModelBase, base_model_class):
        def _get_base_model_class(self):
            return base_model_class

    WaveModel.__name__ = base_model_name
    WaveModel.__qualname__ = base_model_name
    WaveModel.__doc__ = f"Custom {base_model_name} that handles wave feature fusion"

    return WaveModel


def create_wave_causal_model_class(base_model_class: Type,
                                 base_causal_class: Type,
                                 base_model_name: str,
                                 base_causal_name: str) -> Type:
    """
    Factory function to create a wave-enhanced causal language model class

    Args:
        base_model_class: The base model class (e.g., _Phi3Model, _Qwen2Model)
        base_causal_class: The base causal LM class (e.g., _Phi3ForCausalLM, _Qwen2ForCausalLM)
        base_model_name: Name for the base model (e.g., 'Phi3Model')
        base_causal_name: Name for the causal model (e.g., 'Phi3ForCausalLM')

    Returns:
        A new class that inherits from WaveModelForCausalBase and the base causal LM class
    """
    # First create the corresponding wave model
    wave_model_class = create_wave_model_class(base_model_class, base_model_name)

    class WaveCausalModel(WaveModelForCausalBase, base_causal_class):
        def __init__(self, config, **kwargs):
            # Initialize the parent causal LM first
            super(base_causal_class, self).__init__(config, **kwargs)

            # Replace self.model with our custom wave model
            self.model = wave_model_class(config)
            self.post_init()

            # Extract dtype from kwargs for compatibility with newer transformers
            dtype = kwargs.pop('dtype', None)
            if dtype is not None:
                self.to(dtype=dtype)

        def _get_base_model_class(self):
            return base_causal_class

    WaveCausalModel.__name__ = base_causal_name
    WaveCausalModel.__qualname__ = base_causal_name
    WaveCausalModel.__doc__ = f"Custom {base_causal_name} that uses custom {base_model_name}"

    return WaveCausalModel


# Model configuration mapping for easy extension
MODEL_CONFIGS = {
    'phi3': {
        'base_model_class': _Phi3Model,
        'base_causal_class': _Phi3ForCausalLM,
        'model_name': 'Phi3Model',
        'causal_name': 'Phi3ForCausalLM',
    },
    'phi4': {
        'base_model_class': _Phi3Model,  # Phi-4 uses Phi3 architecture
        'base_causal_class': _Phi3ForCausalLM,
        'model_name': 'Phi3Model',  # Use the same base model
        'causal_name': 'Phi3ForCausalLM',
    },
    'qwen2': {
        'base_model_class': _Qwen2Model,
        'base_causal_class': _Qwen2ForCausalLM,
        'model_name': 'Qwen2Model',
        'causal_name': 'Qwen2ForCausalLM',
    },
}


# Dynamically create all model classes
MODEL_CLASSES = {}
WAVE_MODEL_CLASSES = {}

for model_key, config in MODEL_CONFIGS.items():
    # Create wave model class
    wave_model_class = create_wave_model_class(
        config['base_model_class'],
        config['model_name']
    )
    WAVE_MODEL_CLASSES[model_key] = wave_model_class

    # Create wave causal model class
    wave_causal_class = create_wave_causal_model_class(
        config['base_model_class'],
        config['base_causal_class'],
        config['model_name'],
        config['causal_name']
    )
    MODEL_CLASSES[model_key] = wave_causal_class


# Export the created classes with their original names
Phi3Model = WAVE_MODEL_CLASSES['phi3']
Phi3ForCausalLM = MODEL_CLASSES['phi3']
Qwen2Model = WAVE_MODEL_CLASSES['qwen2']
Qwen2ForCausalLM = MODEL_CLASSES['qwen2']

# For phi4, we reuse phi3 classes since they use the same architecture
Phi4ForCausalLM = MODEL_CLASSES['phi4']  # This will be the same as Phi3ForCausalLM


class ModelFactory:
    """Factory class for creating models and tokenizers"""

    MODEL_TYPES = {
        'phi3': Phi3ForCausalLM,
        'phi4': Phi4ForCausalLM,  # Phi-4-mini-instruct uses Phi3 architecture
        'qwen2': Qwen2ForCausalLM,  # Qwen3 uses Qwen2 architecture
    }

    @classmethod
    def create_model(cls, model_type: str, config):
        """
        Create model instance

        Args:
            model_type: 'phi3', 'phi4', or 'qwen2'
            config: Model configuration dict

        Returns:
            Model instance with mmwave support
        """
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load model config
        model_path = config.get('model_path')
        original_config = AutoConfig.from_pretrained(model_path)

        # Update config
        original_config.model_max_length = config.get('model_max_length', 2048)

        # Create model
        model = cls.MODEL_TYPES[model_type].from_pretrained(
            model_path,
            config=original_config,
            dtype=torch.bfloat16,
        )

        return model

    @classmethod
    def create_tokenizer(cls, config):
        """Create tokenizer instance"""
        model_path = config.get('model_path')
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=config.get('model_max_length', 2048),
            padding_side=config.get('padding_side', 'right'),
            use_fast=config.get('use_fast', False),
            local_files_only=config.get('local_files_only', True)
        )

        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type"""
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}")
        return MODEL_CONFIGS[model_type]

    @classmethod
    def register_model(cls, model_key: str, base_model_class: Type,
                      base_causal_class: Type, model_name: str, causal_name: str):
        """
        Register a new model type

        Args:
            model_key: Key for the model (e.g., 'new_model')
            base_model_class: The base model class
            base_causal_class: The base causal LM class
            model_name: Name for the base model
            causal_name: Name for the causal model

        Returns:
            Tuple of (wave_model_class, wave_causal_model_class)
        """
        config = {
            'base_model_class': base_model_class,
            'base_causal_class': base_causal_class,
            'model_name': model_name,
            'causal_name': causal_name,
        }

        MODEL_CONFIGS[model_key] = config

        # Create the classes
        wave_model_class = create_wave_model_class(base_model_class, model_name)
        wave_causal_class = create_wave_causal_model_class(
            base_model_class, base_causal_class, model_name, causal_name
        )

        WAVE_MODEL_CLASSES[model_key] = wave_model_class
        MODEL_CLASSES[model_key] = wave_causal_class
        cls.MODEL_TYPES[model_key] = wave_causal_class

        # Also add to module globals for direct import
        import sys
        current_module = sys.modules[__name__]
        setattr(current_module, causal_name, wave_causal_class)
        setattr(current_module, model_name, wave_model_class)

        return wave_model_class, wave_causal_class