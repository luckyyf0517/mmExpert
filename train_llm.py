"""Training entry point for fine-tuning Phi3 with mmwave features using PyTorch Lightning"""

import argparse
import os
import warnings
from datetime import datetime
from termcolor import colored
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from swanlab.integration.pytorch_lightning import SwanLabLogger

# Ignore SwanLab warning about experiment already in progress
warnings.filterwarnings('ignore', message='.*There is a swanlab experiment already in progress.*')

# Ignore PyTorch Lightning model summary precision warning (bf16-mixed not supported for size estimation)
warnings.filterwarnings('ignore', message='.*Precision bf16-mixed is not supported by the model summary.*')

from easydict import EasyDict
from src.llm.utils.config_loader import load_yaml_config
from src.llm.utils.deepspeed_utils import add_deepspeed_args, get_train_ds_config
from src.llm.utils.common_utils import is_rank_0, log_info, override_config, save_training_artifacts
from src.llm.datamodule import WaveLLMDataModule
from src.llm.trainer import WaveLLMTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune Phi3 with mmwave features")

    # Config
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')

    # Data root path (overrides config file)
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to feature directory (overrides config file)')

    # Training parameters (override config file)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config file)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers (overrides config file)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum number of training epochs (overrides config file)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum number of training steps (overrides max_epochs)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Gradient accumulation steps (overrides config file)')

    # Basic training args
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to display Question/Answer/Prediction for each batch')

    # DeepSpeed args
    parser = add_deepspeed_args(parser)
    
    # Local rank argument (automatically set by DeepSpeed launcher)
    parser.add_argument('--local_rank', type=int, default=None,
                        help='Local rank for distributed training (set by DeepSpeed launcher)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration from YAML
    cfg = load_yaml_config(args.config)

    # Override data_root if provided via command line
    if args.data_root is not None:
        cfg.data_cfg.data_root = args.data_root
        cfg.model_cfg.encoder_path = args.data_root
        log_info(f"Using data_root from command line: {args.data_root}")
    else:
        if not cfg.data_cfg.data_root:
            raise ValueError("data_root must be provided either via --data_root argument or in config file")
        if not cfg.model_cfg.encoder_path:
            cfg.model_cfg.encoder_path = cfg.data_cfg.data_root

    # Override training parameters if provided via command line
    override_config(cfg.data_cfg, 'batch_size', args.batch_size, 'batch_size')
    override_config(cfg.data_cfg, 'num_workers', args.num_workers, 'num_workers')
    override_config(cfg.training, 'max_epochs', args.max_epochs, 'max_epochs')
    override_config(cfg.training, 'gradient_accumulation_steps', args.gradient_accumulation_steps, 'gradient_accumulation_steps')

    # Set seeds
    pl.seed_everything(args.seed)

    # Add timestamp prefix to log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_parent = os.path.dirname(cfg.log_dir)
    log_dir_name = os.path.basename(cfg.log_dir)
    cfg.log_dir = os.path.join(log_dir_parent, f"{timestamp}_{log_dir_name}")
    log_info(f"Using log directory: {cfg.log_dir}")
    
    # Extract experiment name from log_dir (the directory name with timestamp)
    experiment_name = os.path.basename(cfg.log_dir)

    # Initialize data module
    data_module = WaveLLMDataModule(cfg.data_cfg)

    # Create model (LightningModule)
    # Pass model_cfg with training config added
    model_cfg_dict = dict(cfg.model_cfg)
    model_cfg_dict['training'] = EasyDict(cfg.training)
    model_cfg_with_training = EasyDict(model_cfg_dict)
    # Add debug flag to model config
    model_cfg_with_training.debug = args.debug
    model = WaveLLMTrainer(model_cfg_with_training)

    # Setup callbacks - no checkpoint saving (we'll save LoRA adapter manually)
    callbacks = [
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Setup SwanLab logger
    swanlab_logger = SwanLabLogger(name=experiment_name, project='mmExpert-LLM')

    # Create DeepSpeed strategy
    strategy = DeepSpeedStrategy(
        stage=args.zero_stage,
        offload_optimizer=args.offload,
        offload_parameters=args.offload,
        config=get_train_ds_config(args)
    )

    # Create trainer
    # When using deepspeed launcher, devices can be auto-detected from environment
    # But we still need world_size for DeepSpeed config calculation
    trainer_kwargs = {
        'accelerator': 'gpu',
        'devices': "auto",  # Auto-detect from deepspeed launcher
        'strategy': strategy,
        'precision': 'bf16-mixed',
        'accumulate_grad_batches': cfg.training.gradient_accumulation_steps,
        'gradient_clip_val': 1.0,
        'val_check_interval': 1.0,
        'log_every_n_steps': 1,
        'enable_checkpointing': False,  # Disable automatic checkpointing (we save manually)
        'enable_progress_bar': True,
        'deterministic': True,
        'callbacks': callbacks,
        'logger': swanlab_logger,
        'default_root_dir': cfg.log_dir
    }
    
    # Use max_steps if provided, otherwise use max_epochs
    if args.max_steps is not None:
        trainer_kwargs['max_steps'] = args.max_steps
        log_info(f"Using max_steps: {args.max_steps}")
    else:
        trainer_kwargs['max_epochs'] = cfg.training.max_epochs
    
    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    log_info("Starting training...")
    trainer.fit(model, datamodule=data_module)
    log_info("Training completed successfully")
    
    # Save training artifacts (only on rank 0)
    if is_rank_0():
        save_training_artifacts(model, cfg.log_dir)


if __name__ == "__main__":
    main()