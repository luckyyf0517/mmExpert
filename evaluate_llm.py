"""Evaluation script for fine-tuned Phi3 with mmwave features using PyTorch Lightning"""

import torch
import argparse
import os
import random
import numpy as np
from termcolor import colored
import pytorch_lightning as pl
from easydict import EasyDict

from src.llm.utils.config_loader import load_yaml_config
from src.llm.datamodule import WaveLLMDataModule
from src.llm.trainer import WaveLLMTrainer

# Import shared utility functions
from src.llm.utils.common_utils import override_config, load_model_from_checkpoint
from src.logger import log_message


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser.add_argument("--use_random_question", type=lambda x: x.lower() == 'true', default=None,
                        help="Whether to use random question selection (true/false, overrides config file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory for evaluation results (default: evaluation)")

    # Generate parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K for sampling (default: 50)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-P for sampling (default: 0.95)")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for beam search (default: 4)")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Maximum number of new tokens to generate (default: 30)")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Length penalty for beam search (default: 1.0)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty (default: 1.0)")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="Whether to use sampling (default: True)")
    parser.add_argument("--early_stopping", action="store_true", default=False,
                        help="Whether to use early stopping in beam search (default: False)")

    # Local rank argument (automatically set by DeepSpeed launcher)
    parser.add_argument("--local_rank", type=int, default=None,
                        help="Local rank for distributed evaluation (set by DeepSpeed launcher)")

    args = parser.parse_args()
    
    # Set local_rank to environment variable if provided (for DeepSpeed launcher compatibility)
    if args.local_rank is not None:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # Set seeds
    set_global_seed(args.seed)

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
        # If split doesn't contain "QA", set caption_only to true
        if "QA" not in args.split:
            override_config(cfg.data_cfg, 'caption_only', True, 'caption_only')
            log_message("CONFIG", f"Split '{args.split}' doesn't contain 'QA', setting caption_only=True", color="blue")
    if args.batch_size is not None:
        override_config(cfg.data_cfg, 'batch_size', args.batch_size, 'batch_size')
    override_config(cfg.data_cfg, 'use_random_question', args.use_random_question, 'use_random_question')

    # Add generate parameters to config
    cfg.generation = EasyDict({
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'num_beams': args.num_beams,
        'max_new_tokens': args.max_new_tokens,
        'length_penalty': args.length_penalty,
        'repetition_penalty': args.repetition_penalty,
        'do_sample': args.do_sample,
        'early_stopping': args.early_stopping,
    })

    # Log generation parameters
    log_message("CONFIG", f"Generation parameters:", color="blue")
    log_message("CONFIG", f"  temperature: {cfg.generation.temperature}", color="blue")
    log_message("CONFIG", f"  top_k: {cfg.generation.top_k}", color="blue")
    log_message("CONFIG", f"  top_p: {cfg.generation.top_p}", color="blue")
    log_message("CONFIG", f"  num_beams: {cfg.generation.num_beams}", color="blue")
    log_message("CONFIG", f"  max_new_tokens: {cfg.generation.max_new_tokens}", color="blue")
    log_message("CONFIG", f"  length_penalty: {cfg.generation.length_penalty}", color="blue")
    log_message("CONFIG", f"  repetition_penalty: {cfg.generation.repetition_penalty}", color="blue")
    log_message("CONFIG", f"  do_sample: {cfg.generation.do_sample}", color="blue")
    log_message("CONFIG", f"  early_stopping: {cfg.generation.early_stopping}", color="blue")
    
    # Set default batch_size if not in config
    if not hasattr(cfg.data_cfg, 'batch_size') or cfg.data_cfg.batch_size is None:
        cfg.data_cfg.batch_size = 4  # Default batch size for evaluation
        log_message("CONFIG", f"Using default batch_size: {cfg.data_cfg.batch_size}", color="blue")

    # Set output directory
    if args.output_dir is not None:
        cfg.evaluation_dir = args.output_dir
        log_message("CONFIG", f"Using custom evaluation directory: {cfg.evaluation_dir}", color="blue")
    else:
        cfg.evaluation_dir = "evaluation"
        log_message("CONFIG", f"Using default evaluation directory: {cfg.evaluation_dir}", color="blue")

    # CRITICAL: Ensure the model uses the updated config
    model.cfg = cfg
    log_message("CONFIG", f"Model cfg updated with evaluation_dir: {cfg.evaluation_dir}", color="blue")

    # Load dataset using DataModule (same as training)
    log_message("LOAD", f"Loading dataset from {cfg.data_cfg.data_root}", color="cyan")

    # Set inference mode for evaluation
    cfg.data_cfg.is_inference = True
    log_message("CONFIG", f"Setting inference mode: is_inference=True", color="blue")

    data_module = WaveLLMDataModule(cfg.data_cfg)
    data_module.setup(stage='test')

    # Create trainer for testing
    # Support distributed evaluation when launched with DeepSpeed
    # Use DDP strategy for evaluation (no need for DeepSpeed ZeRO optimization)
    trainer_kwargs = {
        'accelerator': 'gpu',
        'devices': "auto",  # Auto-detect from deepspeed launcher
        'strategy': 'ddp',  # Use DDP for distributed evaluation
        'precision': 'bf16-mixed',
        'enable_progress_bar': True,
        'default_root_dir': checkpoint_path
    }
    
    # If not using DeepSpeed launcher, fall back to single GPU
    if args.local_rank is None and os.environ.get('WORLD_SIZE') is None:
        trainer_kwargs['devices'] = 1
        trainer_kwargs['strategy'] = 'auto'
    
    trainer = pl.Trainer(**trainer_kwargs)

    # Run test
    log_message("PROGRESS", "Starting evaluation...", color="yellow")
    trainer.test(model, datamodule=data_module)
    
    log_message("SUCCESS", "Evaluation completed! Results saved to evaluation directory", color="green")


if __name__ == "__main__":
    main()