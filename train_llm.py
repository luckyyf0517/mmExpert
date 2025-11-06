"""Training entry point for fine-tuning Phi3 with mmwave features using PyTorch Lightning"""

import argparse
import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from swanlab.integration.pytorch_lightning import SwanLabLogger
import swanlab

from easydict import EasyDict
from src.llm.utils.config_loader import load_yaml_config
from src.llm.utils.deepspeed_utils import add_deepspeed_args, get_train_ds_config
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
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Gradient accumulation steps (overrides config file)')

    # Basic training args
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (no training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

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
        # Update both data_cfg.data_root and model_cfg.encoder_path
        cfg.data_cfg.data_root = args.data_root
        cfg.model_cfg.encoder_path = args.data_root
        print(f"Using data_root from command line: {args.data_root}")
    else:
        # Check if data_root is set in config file
        if not cfg.data_cfg.data_root or cfg.data_cfg.data_root == "":
            raise ValueError("data_root must be provided either via --data_root argument or in config file")
        # Update encoder_path to match data_root if not already set
        if not cfg.model_cfg.encoder_path or cfg.model_cfg.encoder_path == "":
            cfg.model_cfg.encoder_path = cfg.data_cfg.data_root

    # Override training parameters if provided via command line, otherwise use config file values
    if args.batch_size is not None:
        cfg.data_cfg.batch_size = args.batch_size
        print(f"Using batch_size from command line: {args.batch_size}")
    elif not hasattr(cfg.data_cfg, 'batch_size') or cfg.data_cfg.batch_size is None:
        raise ValueError("batch_size must be provided either via --batch_size argument or in config file")
    
    if args.num_workers is not None:
        cfg.data_cfg.num_workers = args.num_workers
        print(f"Using num_workers from command line: {args.num_workers}")
    elif not hasattr(cfg.data_cfg, 'num_workers') or cfg.data_cfg.num_workers is None:
        raise ValueError("num_workers must be provided either via --num_workers argument or in config file")
    
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs
        print(f"Using max_epochs from command line: {args.max_epochs}")
    elif not hasattr(cfg.training, 'max_epochs') or cfg.training.max_epochs is None:
        raise ValueError("max_epochs must be provided either via --max_epochs argument or in config file")
    
    if args.gradient_accumulation_steps is not None:
        cfg.training.gradient_accumulation_steps = args.gradient_accumulation_steps
        print(f"Using gradient_accumulation_steps from command line: {args.gradient_accumulation_steps}")
    elif not hasattr(cfg.training, 'gradient_accumulation_steps') or cfg.training.gradient_accumulation_steps is None:
        raise ValueError("gradient_accumulation_steps must be provided either via --gradient_accumulation_steps argument or in config file")

    # Set seeds
    pl.seed_everything(args.seed)

    # Add timestamp prefix to log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_path = cfg.log_dir
    log_dir_parent = os.path.dirname(log_dir_path)
    log_dir_name = os.path.basename(log_dir_path)
    log_dir_with_timestamp = os.path.join(log_dir_parent, f"{timestamp}_{log_dir_name}")
    cfg.log_dir = log_dir_with_timestamp
    print(f"Using log directory: {cfg.log_dir}")
    
    # Extract experiment name from log_dir (the directory name with timestamp)
    experiment_name = os.path.basename(log_dir_with_timestamp)

    # Initialize data module
    data_module = WaveLLMDataModule(cfg.data_cfg)

    # Create model (LightningModule)
    # Pass model_cfg with training config added
    model_cfg_dict = dict(cfg.model_cfg)
    model_cfg_dict['training'] = EasyDict(cfg.training)
    model_cfg_with_training = EasyDict(model_cfg_dict)
    model = WaveLLMTrainer(model_cfg_with_training)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.log_dir,
            filename="model-epoch={epoch:02d}-val_loss={valid/loss:.4f}",
            monitor="valid/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Setup SwanLab logger
    # Ensure previous SwanLab experiment is finished before creating new logger
    import time
    try:
        swanlab.finish()
        # Wait a moment to ensure the previous experiment is fully closed
        time.sleep(0.5)
    except Exception:
        # If finish fails, try to clear any existing experiment
        try:
            swanlab.finish()
            time.sleep(0.5)
        except Exception:
            pass
    
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
    trainer = pl.Trainer(
        accelerator='gpu',
        devices="auto",  # Auto-detect from deepspeed launcher
        strategy=strategy,
        precision='bf16-mixed',
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=True,
        callbacks=callbacks,
        logger=swanlab_logger,
        default_root_dir=cfg.log_dir
    )

    # Train or test
    if not args.test:
        print("🚀 Starting training...")
        trainer.fit(model, datamodule=data_module)
        print("✅ Training completed successfully")
    else:
        print("🧪 Running test...")
        trainer.test(model, datamodule=data_module)
        print("✅ Testing completed successfully")


if __name__ == "__main__":
    main()