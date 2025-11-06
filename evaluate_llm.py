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

# Import shared utility functions
from src.llm.utils.common_utils import is_rank_0, log_info, override_config, load_and_verify_projection_layers, load_model_from_checkpoint


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