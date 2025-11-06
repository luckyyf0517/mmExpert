"""Evaluation script for fine-tuned Phi3 with mmwave features using PyTorch Lightning"""

import argparse
import os
from termcolor import colored
import pytorch_lightning as pl
from easydict import EasyDict

from src.llm.utils.config_loader import load_yaml_config
from src.llm.datamodule import WaveLLMDataModule
from src.llm.trainer import WaveLLMTrainer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

# Import shared functions from train_llm
from train_llm import is_rank_0, log_info, override_config


def load_model_from_checkpoint(checkpoint_path, config_path, data_root=None):
    """Load model from checkpoint (supports both Lightning and DeepSpeed checkpoints)"""
    log_info(f"Loading model from checkpoint: {checkpoint_path}")
    
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

    # Check if it's a DeepSpeed checkpoint (directory with checkpoint/ subdirectory)
    is_deepspeed_checkpoint = os.path.isdir(checkpoint_path) and \
                              os.path.exists(os.path.join(checkpoint_path, 'checkpoint'))
    
    if is_deepspeed_checkpoint:
        # For DeepSpeed checkpoint, create model first, then load state dict
        log_info("Detected DeepSpeed checkpoint, using load_state_dict_from_zero_checkpoint")
        model = WaveLLMTrainer(model_cfg_with_training)
        # Load state dict from DeepSpeed ZeRO checkpoint
        model = load_state_dict_from_zero_checkpoint(model, checkpoint_path)
    else:
        # For regular Lightning checkpoint, use load_from_checkpoint
        model = WaveLLMTrainer.load_from_checkpoint(
            checkpoint_path,
            cfg=model_cfg_with_training,
            strict=False  # Allow missing keys (e.g., optimizer state)
        )
    
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
    parser = argparse.ArgumentParser(description="Evaluate mmwave+LLM model")

    # Required arguments
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file (same as training)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data directory (overrides config file)")

    # Optional arguments
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to evaluate (overrides config file)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for evaluation (overrides config file)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum new tokens to generate (overrides config)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Generation temperature (overrides config)")

    args = parser.parse_args()

    # Check and resolve checkpoint path
    checkpoint_path = args.model_checkpoint
    if not os.path.exists(checkpoint_path):
        print(colored("[ERROR]", "red") + f" Model checkpoint not found: {args.model_checkpoint}")
        return

    # PyTorch Lightning supports both file and directory checkpoints
    # - File: regular .ckpt file
    # - Directory: DeepSpeed checkpoint (contains checkpoint/ subdirectory)
    if os.path.isdir(checkpoint_path):
        # Check if it's a DeepSpeed checkpoint directory
        checkpoint_subdir = os.path.join(checkpoint_path, 'checkpoint')
        if os.path.exists(checkpoint_subdir):
            log_info(f"Detected DeepSpeed checkpoint directory: {checkpoint_path}")
        else:
            # If it's a directory but not DeepSpeed format, look for .ckpt file inside
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
            if ckpt_files:
                checkpoint_path = os.path.join(checkpoint_path, ckpt_files[0])
                log_info(f"Found checkpoint file in directory: {checkpoint_path}")
            else:
                print(colored("[ERROR]", "red") + f" Invalid checkpoint directory: {args.model_checkpoint}")
                print(colored("[INFO]", "yellow") + " Expected either a .ckpt file or a DeepSpeed checkpoint directory")
                return
    else:
        log_info(f"Using checkpoint file: {checkpoint_path}")

    # Check if config file exists
    if not os.path.exists(args.config):
        print(colored("[ERROR]", "red") + f" Config file not found: {args.config}")
        return

    # Load model from checkpoint
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

    # Override test generation parameters
    if args.max_new_tokens is not None:
        cfg.test_max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.test_temperature = args.temperature

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
        default_root_dir=os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path
    )

    # Run test
    log_info("Starting evaluation...")
    trainer.test(model, datamodule=data_module)
    
    log_info("Evaluation completed! Results saved to evaluation directory")


if __name__ == "__main__":
    main()