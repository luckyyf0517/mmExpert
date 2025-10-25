import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import yaml
import glob
import shutil
import random
import argparse
import logging
import numpy as np
from datetime import datetime

import sys
sys.path.append('.')

import torch
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")
warnings.filterwarnings('ignore', '.*find_unused_parameters=True was specified.*')
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
warnings.filterwarnings('ignore', '.*0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives.*')
warnings.filterwarnings("ignore", ".*The pynvml package is deprecated. Please install nvidia-ml-py instead.*")


# Ignore specific PyTorch Lightning warning
warnings.filterwarnings("ignore", message="It is recommended to use `self.log('valid/loss_all', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.")

from pytorch_lightning.utilities import disable_possible_user_warnings
disable_possible_user_warnings()

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from swanlab.integration.pytorch_lightning import SwanLabLogger
import swanlab

from src.misc.io import load_config
from src.misc.tools import instantiate_from_config
from src.misc.config_printer import print_core_config


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-checkpoint", default=None, type=str, required=False)
    parser.add_argument("--version", '-v', default=None, type=str, required=False)
    parser.add_argument('--seed', dest="seed", default=88, type=int, help="random seed")
    parser.add_argument('--test', dest="test", action="store_true", default=False)

    # Add configuration control arguments
    parser.add_argument('--log-dir', dest="log_dir", default='log/', type=str, help="log directory")
    parser.add_argument('--strategy', dest="strategy", default='ddp', type=str, help="training strategy")

    # Allow separate data and model config files
    parser.add_argument('--data-config', dest="data_config", default="config/data/humanml3d.yaml", type=str, help="data config file path")
    parser.add_argument('--model-config', dest="model_config", default=None, type=str, help="model config file path")

    # Add dry run mode
    parser.add_argument('--dry-run', dest="dry_run", action="store_true", default=False,
                       help="Dry run mode: run only 1 train/val step, no logging, for config validation")

    args = parser.parse_args()
    if args.test:
        assert args.resume_checkpoint is not None

    args.rank = int(os.environ.get('RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    return args


if __name__ == '__main__':
    args = parse_args()

    # Load configurations using args
    cfg = load_config(None, args.data_config, args.model_config)

    # Extract model name only (basename without extension)
    model_name = os.path.splitext(os.path.basename(args.model_config))[0]

    # Extract dataset name from data config path
    data_config_name = os.path.splitext(os.path.basename(args.data_config))[0]

    # Dry run mode handling
    if args.dry_run:
        print("🧪 DRY RUN MODE - Configuration Validation")
        print("=" * 50)
        print(f"📋 Model Config: {args.model_config}")
        print(f"📊 Data Config: {args.data_config}")
        print(f"🎯 Strategy: {args.strategy}")
        print(f"🌱 Seed: {args.seed}")
        print("=" * 50)

        # Validate configs without logging
        try:
            # Validate model config
            model_cfg = cfg.model_cfg
            print("✅ Model configuration loaded successfully")
            print(f"   - Model type: {model_cfg.target}")
            print(f"   - Max epochs: {model_cfg.params.max_epochs}")
            print(f"   - Learning rate: {model_cfg.params.learning_rate}")
            print(f"   - Temperature: {model_cfg.params.temperature}")

            # Validate data config
            data_cfg = cfg.data_cfg
            print("✅ Data configuration loaded successfully")
            print(f"   - Dataset: {data_cfg.target}")
            print(f"   - Batch size: {data_cfg.params.cfg.batch_size}")

            # Test model instantiation (minimal)
            print("✅ Testing model instantiation...")
            model = instantiate_from_config(model_cfg)
            print(f"   - Model instantiated: {type(model).__name__}")

            # Test data loading (minimal)
            print("✅ Testing data loading...")
            data = instantiate_from_config(data_cfg)
            print(f"   - Data module instantiated: {type(data).__name__}")

            # Test forward pass with minimal data
            print("✅ Testing forward pass...")
            data.setup('fit')

            # Get single batch for testing
            train_loader = data.train_dataloader()
            val_loader = data.val_dataloader()

            print("   - Loading single batch from train loader...")
            train_batch = next(iter(train_loader))
            train_shapes = {k: (v.shape if hasattr(v, 'shape') else str(type(v))) for k, v in train_batch.items()}
            print(f"   - Train batch shapes: {train_shapes}")

            print("   - Loading single batch from val loader...")
            val_batch = next(iter(val_loader))
            val_shapes = {k: (v.shape if hasattr(v, 'shape') else str(type(v))) for k, v in val_batch.items()}
            print(f"   - Val batch shapes: {val_shapes}")

            # Test model forward pass
            print("   - Running model forward pass...")
            with torch.no_grad():
                # Dummy step for validation
                dummy_logits = model(train_batch)
                if hasattr(dummy_logits, 'loss'):
                    print(f"   - Model output (loss): {dummy_logits.loss:.6f}")
                else:
                    output_shapes = {k: (v.shape if hasattr(v, 'shape') else str(type(v))) for k, v in dummy_logits.items() if hasattr(v, 'shape')}
                    print(f"   - Model output shape: {output_shapes}")

            print("\n🎉 DRY RUN SUCCESSFUL - Configuration is valid!")
            print("💡 Use this command to start full training:")
            print(f"   python run_clip.py --model-config {args.model_config} --data-config {args.data_config}")

        except Exception as e:
            print(f"\n❌ DRY RUN FAILED - Configuration error detected!")
            print(f"Error: {str(e)}")
            print("\n🔧 Please check your configuration files:")
            print(f"   - Model config: {args.model_config}")
            print(f"   - Data config: {args.data_config}")

        exit()

    # Normal training mode (original logic)
    # Extract experiment folder name from model config path
    config_path_parts = args.model_config.split('/')
    exp_folder = None
    for i, part in enumerate(config_path_parts):
        if part.startswith('experiments-'):
            exp_folder = part
            break

    if exp_folder is None:
        exp_folder = 'experiments'

    if args.version is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.version = f"{timestamp}_{model_name}"  # Put timestamp first for better sorting

    # Create log directory structure: dataset_experiment-folder/model_version
    exp_category = f"{data_config_name}_{exp_folder}"
    log_dir = os.path.join(args.log_dir, exp_category, args.version)
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    config_dir = os.path.join(log_dir, 'config')  # Config files now in log directory

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Save config files for reference
    if args.data_config is not None and args.model_config is not None:
        shutil.copy(args.data_config, os.path.join(config_dir, 'data_config.yaml'))
        shutil.copy(args.model_config, os.path.join(config_dir, 'model_config.yaml'))
    else:
        raise ValueError("Data-config and model-config must be provided")

    # Instantiate data module
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.batch_size = data_cfg.params.cfg.batch_size // args.world_size  # for each gpu
    data = instantiate_from_config(data_cfg)
    data.setup('fit')

    # Set random seed
    set_seed(seed=args.seed, n_gpu=args.world_size)

    # Print core configuration parameters (only on rank 0)
    if args.rank == 0:
        print_core_config(args, log_dir, cfg)

    # Instantiate model
    model_cfg = cfg.model_cfg
    model = instantiate_from_config(model_cfg)

    # End previous SwanLab experiment (if exists)
    try:
        swanlab.finish()
    except:
        pass

    logger = SwanLabLogger(name=args.version, project='mmExpert')
    if not args.resume_checkpoint and args.rank == 0:
        for log_file in glob.glob(os.path.join(checkpoints_dir, '*.ckpt')):
            os.remove(log_file)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor='valid/loss_clip',
        filename='epoch_{epoch:02d}_val_{valid/loss_clip:.4f}',
        save_top_k=10,
        mode='min',
        auto_insert_metric_name=False,
        save_last=True,
        save_weights_only=False,)

    trainer = Trainer(
        accelerator='gpu',
        devices=args.world_size,
        strategy=args.strategy,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=cfg.model_cfg.params.max_epochs,
        num_sanity_val_steps=2, # run validation step experimentaly
        reload_dataloaders_every_n_epochs=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True)

    if not args.test:
        trainer.fit(model, train_dataloaders=data.train_dataloader(),
                    val_dataloaders=[data.val_dataloader()],
                    ckpt_path=args.resume_checkpoint if args.resume_checkpoint else None)
    else:
        trainer.test(model, datamodule=data, ckpt_path=args.resume_checkpoint if args.resume_checkpoint else None)
