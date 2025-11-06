"""Evaluation script for fine-tuned Phi3 with mmwave features using PyTorch Lightning"""

import torch
import argparse
import os
from termcolor import colored
import pytorch_lightning as pl
from easydict import EasyDict

from src.llm.utils.config_loader import load_yaml_config
from src.llm.datamodule import WaveLLMDataModule
from src.llm.trainer import WaveLLMTrainer

# Import shared functions from train_llm
from train_llm import is_rank_0, log_info, override_config


def load_and_verify_projection_layers(model, checkpoint_path):
    """Load and verify projection layers from checkpoint
    
    Args:
        model: WaveLLMTrainer model instance
        checkpoint_path: Path to checkpoint directory containing non_lora_trainables.safetensors
        
    Returns:
        None (raises exception if loading fails)
    """
    from safetensors.torch import load_file
    
    # Load non-LoRA trainables (mm_projection_layers)
    projection_path = os.path.join(checkpoint_path, "non_lora_trainables.safetensors")
    if not os.path.exists(projection_path):
        raise FileNotFoundError(f"Projection layers file not found: {projection_path}")
    
    log_info(f"Loading projection layers from {projection_path}")
    non_lora_state = load_file(projection_path)
    
    # Verify that we have the expected keys
    expected_keys = {'mm_projection_layers.weight', 'mm_projection_layers.bias'}
    actual_keys = set(non_lora_state.keys())
    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys
    
    if missing_keys:
        raise ValueError(f"Missing expected keys in projection layers: {missing_keys}")
    if unexpected_keys:
        log_info(f"Warning: Unexpected keys in projection layers (will be ignored): {unexpected_keys}")
    
    # Load projection layers directly (cleaner than load_state_dict with strict=False)
    projection_state = {
        'weight': non_lora_state['mm_projection_layers.weight'],
        'bias': non_lora_state['mm_projection_layers.bias']
    }
    
    try:
        model.mm_projection_layers.load_state_dict(projection_state)
        log_info("Projection layers loaded successfully")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load projection layers: {e}") from e


def load_model_from_checkpoint(checkpoint_path, config_path, data_root=None):
    """Load model from LoRA adapter checkpoint"""
    log_info(f"Loading model from LoRA adapter: {checkpoint_path}")
    
    # Load configuration
    cfg = load_yaml_config(config_path)
    
    # Override data_root if provided
    if data_root:
        cfg.data_cfg.data_root = data_root
        cfg.model_cfg.encoder_path = data_root
    
    # Prepare model config (same as training)
    model_cfg_dict = dict(cfg.model_cfg)
    model_cfg_dict['training'] = EasyDict(cfg.training)
    model_cfg_with_training = EasyDict(model_cfg_dict)

    # Check if it's a LoRA adapter directory (has adapter_model.safetensors and adapter_config.json)
    has_adapter = os.path.exists(os.path.join(checkpoint_path, 'adapter_model.safetensors'))
    has_config = os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json'))
    
    is_lora_adapter = os.path.isdir(checkpoint_path) and has_adapter and has_config
    
    if not is_lora_adapter:
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}\n"
                        f"Expected LoRA adapter directory with adapter_model.safetensors and adapter_config.json")
    
    # Load from LoRA adapter
    log_info("Detected LoRA adapter directory")
    
    # Temporarily disable LoRA in config to create base model without PEFT wrapping
    original_use_peft = model_cfg_with_training.use_peft
    model_cfg_with_training.use_peft = False
    
    # Create base model (without LoRA)
    model = WaveLLMTrainer(model_cfg_with_training)
    
    # Restore original config
    model_cfg_with_training.use_peft = original_use_peft
    
    # Load LoRA adapter onto base model
    from peft import PeftModel
    import warnings
    
    log_info(f"Loading LoRA adapter from {checkpoint_path}")
    
    # Suppress PEFT warning about missing adapter keys (these are expected when filtering embeddings)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Found missing adapter keys.*')
        # PEFT supports safetensors directly, just use from_pretrained
        model.model = PeftModel.from_pretrained(model.model, checkpoint_path)
    
    # Load and verify projection layers using shared function
    load_and_verify_projection_layers(model, checkpoint_path)
    
    model.eval()
    # Move entire model to GPU
    model = model.cuda()
    # Explicitly move radar_encoder and projection layers to GPU
    if hasattr(model, 'radar_encoder'):
        model.radar_encoder = model.radar_encoder.cuda()
    if hasattr(model, 'mm_projection_layers'):
        model.mm_projection_layers = model.mm_projection_layers.cuda()

    # Set tokenizer padding_side to 'left' for decoder-only generation
    model.tokenizer.padding_side = 'left'
    
    log_info("Model loaded successfully")
    return model, cfg


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate mmwave+LLM model with LoRA adapter")

    # Required arguments
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to LoRA adapter directory (contains adapter_model.bin)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file (same as training)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data directory (overrides config file)")

    # Optional arguments
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to evaluate (overrides config file)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for evaluation (overrides config file)")

    args = parser.parse_args()

    # Check checkpoint path
    checkpoint_path = args.model_checkpoint
    if not os.path.exists(checkpoint_path):
        print(colored("[ERROR]", "red") + f" Checkpoint path not found: {checkpoint_path}")
        return
    
    if not os.path.isdir(checkpoint_path):
        print(colored("[ERROR]", "red") + f" Checkpoint path must be a directory: {checkpoint_path}")
        return

    # Check if config file exists
    if not os.path.exists(args.config):
        print(colored("[ERROR]", "red") + f" Config file not found: {args.config}")
        return

    # Load model from LoRA adapter
    model, cfg = load_model_from_checkpoint(checkpoint_path, args.config, args.data_root)

    # Override config with command line arguments
    if args.split is not None:
        cfg.data_cfg.test_split = args.split
    if args.batch_size is not None:
        override_config(cfg.data_cfg, 'batch_size', args.batch_size, 'batch_size')
    
    # Set default batch_size if not in config
    if not hasattr(cfg.data_cfg, 'batch_size') or cfg.data_cfg.batch_size is None:
        cfg.data_cfg.batch_size = 4  # Default batch size for evaluation
        log_info(f"Using default batch_size: {cfg.data_cfg.batch_size}")

    # Load dataset using DataModule (same as training)
    log_info(f"Loading dataset from {cfg.data_cfg.data_root}")
    data_module = WaveLLMDataModule(cfg.data_cfg)
    data_module.setup(stage='test')

    # Create trainer for testing
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision='bf16-mixed',
        enable_progress_bar=True,
        default_root_dir=checkpoint_path
    )

    # Run test
    log_info("Starting evaluation...")
    trainer.test(model, datamodule=data_module)
    
    log_info("Evaluation completed! Results saved to evaluation directory")


if __name__ == "__main__":
    main()