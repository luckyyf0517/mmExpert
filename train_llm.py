"""Training entry point for fine-tuning Phi3 with mmwave features using PyTorch Lightning"""

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

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

    # Basic training args
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (no training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # DeepSpeed args
    parser = add_deepspeed_args(parser)

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

    # Set seeds
    pl.seed_everything(args.seed)

    # Initialize data module
    data_module = WaveLLMDataModule(cfg.data_cfg)

    # Create model (LightningModule)
    model = WaveLLMTrainer(cfg.model_cfg)

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

    # Create DeepSpeed strategy
    strategy = DeepSpeedStrategy(
        stage=args.zero_stage,
        offload_optimizer=args.offload,
        offload_parameters=args.offload,
        config=get_train_ds_config(args)
    )

    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.world_size,
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