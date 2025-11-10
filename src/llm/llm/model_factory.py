"""Factory class for creating models and tokenizers"""

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from .modeling_phi3 import Phi3ForCausalLM

# TODO: Create qwen3_model.py or import from appropriate location
try:
    from .qwen3_model import Qwen2ForCausalLM
except ImportError:
    # Fallback: use AutoModelForCausalLM for qwen2 if qwen3_model doesn't exist
    Qwen2ForCausalLM = None


class ModelFactory:
    """Factory class for creating models and tokenizers"""

    MODEL_TYPES = {
        'phi3': Phi3ForCausalLM,
        'phi4': Phi3ForCausalLM,  # Phi-4-mini-instruct uses Phi3 architecture
        'qwen2': Qwen2ForCausalLM if Qwen2ForCausalLM is not None else None,  # Qwen3 uses Qwen2 architecture
    }

    @classmethod
    def create_model(cls, model_type: str, config):
        """
        Create model instance

        Args:
            model_type: 'phi3', 'phi4', or 'qwen2'
            config: Model configuration dict

        Returns:
            Model instance (Phi3ForCausalLM or Qwen2ForCausalLM with mmwave support)
        """
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class = cls.MODEL_TYPES[model_type]
        if model_class is None:
            raise ValueError(f"Model type {model_type} is not available. Please implement the corresponding model class.")

        # Load model config
        model_path = config.get('model_path')
        original_config = AutoConfig.from_pretrained(model_path)

        # Update config
        original_config.model_max_length = config.get('model_max_length', 2048)

        # Create model
        model = model_class.from_pretrained(
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