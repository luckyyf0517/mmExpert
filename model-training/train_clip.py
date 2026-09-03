import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_LOADING_REPORT"] = "1"

import argparse
import glob
import random
import shutil
import sys
from datetime import datetime

import numpy as np
from src.logger import log_message

sys.path.append(".")

import torch

torch.set_float32_matmul_precision("high")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")
warnings.filterwarnings("ignore", ".*find_unused_parameters=True was specified.*")
warnings.filterwarnings("ignore", ".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", ".*0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives.*")
warnings.filterwarnings("ignore", ".*The pynvml package is deprecated. Please install nvidia-ml-py instead.*")


# Ignore specific PyTorch Lightning warning
warnings.filterwarnings(
    "ignore",
    message="It is recommended to use `self.log('valid/loss_all', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.",
)

from pytorch_lightning.utilities import disable_possible_user_warnings

disable_possible_user_warnings()

import swanlab
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from src.clip.utils.config_printer import print_core_config
from src.clip.utils.io import load_config
from src.clip.utils.tools import instantiate_from_config
from swanlab.integration.pytorch_lightning import SwanLabLogger


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-checkpoint", default=None, type=str, required=False)
    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        type=str,
        required=False,
        help="Load encoder weights from a checkpoint (weights only, no optimizer/epoch restoration)",
    )
    parser.add_argument("--version", "-v", default=None, type=str, required=False)
    parser.add_argument("--seed", dest="seed", default=88, type=int, help="random seed")

    # Add configuration control arguments
    parser.add_argument("--log-dir", dest="log_dir", default="log/", type=str, help="log directory")
    parser.add_argument("--strategy", dest="strategy", default="ddp", type=str, help="training strategy")

    # Allow separate data and model config files
    parser.add_argument(
        "--data-config",
        dest="data_config",
        default="config/data/humanml3d.yaml",
        type=str,
        help="data config file path",
    )
    parser.add_argument("--model-config", dest="model_config", default=None, type=str, help="model config file path")

    # Add dry run mode
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Dry run mode: run only 1 train/val step, no logging, for config validation",
    )
    parser.add_argument(
        "--save-top-k",
        dest="save_top_k",
        default=10,
        type=int,
        help="Number of best checkpoints to keep (default: 10). Set to 1 to only keep the latest.",
    )

    args = parser.parse_args()
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    return args


if __name__ == "__main__":
    args = parse_args()

    # Load configurations using args
    cfg = load_config(None, args.data_config, args.model_config)

    # Extract model name only (basename without extension)
    model_name = os.path.splitext(os.path.basename(args.model_config))[0]

    # Extract dataset name from data config path
    data_config_name = os.path.splitext(os.path.basename(args.data_config))[0]

    # Dry run mode handling
    if args.dry_run:
        log_message("DRY RUN", "DRY RUN MODE - Configuration Validation", color="cyan")
        print("=" * 50)
        log_message("CONFIG", f"Model Config: {args.model_config}", color="blue")
        log_message("CONFIG", f"Data Config: {args.data_config}", color="blue")
        log_message("CONFIG", f"Strategy: {args.strategy}", color="blue")
        log_message("CONFIG", f"Seed: {args.seed}", color="blue")
        print("=" * 50)

        # Validate configs without logging
        try:
            # Validate model config
            model_cfg = cfg.model_cfg
            log_message("INFO", "Model configuration loaded successfully", color="green")
            log_message("INFO", f"   - Model type: {model_cfg.target}", color="green")
            log_message("INFO", f"   - Max epochs: {model_cfg.params.max_epochs}", color="green")
            log_message("INFO", f"   - Learning rate: {model_cfg.params.learning_rate}", color="green")
            log_message("INFO", f"   - Temperature: {model_cfg.params.temperature}", color="green")

            # Validate data config
            data_cfg = cfg.data_cfg
            log_message("INFO", "Data configuration loaded successfully", color="green")
            log_message("INFO", f"   - Dataset: {data_cfg.target}", color="green")
            log_message("INFO", f"   - Batch size: {data_cfg.params.cfg.batch_size}", color="green")

            # Test model instantiation (minimal)
            log_message("INFO", "Testing model instantiation...", color="green")
            model = instantiate_from_config(model_cfg)
            log_message("INFO", f"   - Model instantiated: {type(model).__name__}", color="green")

            # Test data loading (minimal)
            log_message("INFO", "Testing data loading...", color="green")
            data = instantiate_from_config(data_cfg)
            log_message("INFO", f"   - Data module instantiated: {type(data).__name__}", color="green")

            # Test forward pass with minimal data
            log_message("INFO", "Testing forward pass...", color="green")
            data.setup("fit")

            # Get single batch for testing
            train_loader = data.train_dataloader()
            val_loader = data.val_dataloader()

            log_message("INFO", "   - Loading single batch from train loader...", color="green")
            train_batch = next(iter(train_loader))
            train_shapes = {k: (v.shape if hasattr(v, "shape") else str(type(v))) for k, v in train_batch.items()}
            log_message("INFO", f"   - Train batch shapes: {train_shapes}", color="green")

            log_message("INFO", "   - Loading single batch from val loader...", color="green")
            val_batch = next(iter(val_loader))
            val_shapes = {k: (v.shape if hasattr(v, "shape") else str(type(v))) for k, v in val_batch.items()}
            log_message("INFO", f"   - Val batch shapes: {val_shapes}", color="green")

            # Test model forward pass
            log_message("INFO", "   - Running model forward pass...", color="green")
            with torch.no_grad():
                # Dummy step for validation
                dummy_logits = model(
                    radar_data=train_batch["radar_data"],
                    text=train_batch["caption"],
                )
                if hasattr(dummy_logits, "loss"):
                    log_message("INFO", f"   - Model output (loss): {dummy_logits.loss:.6f}", color="green")
                elif isinstance(dummy_logits, dict):
                    output_shapes = {
                        k: (v.shape if hasattr(v, "shape") else str(type(v)))
                        for k, v in dummy_logits.items()
                        if hasattr(v, "shape")
                    }
                    log_message("INFO", f"   - Model output shape: {output_shapes}", color="green")
                elif isinstance(dummy_logits, tuple):
                    tuple_shapes = [(v.shape if hasattr(v, "shape") else str(type(v))) for v in dummy_logits]
                    log_message("INFO", f"   - Model output tuple: {tuple_shapes}", color="green")
                else:
                    log_message("INFO", f"   - Model output type: {type(dummy_logits).__name__}", color="green")

            log_message("SUCCESS", "\nDRY RUN SUCCESSFUL - Configuration is valid!", color="green")
            log_message("INFO", "Use this command to start full training:", color="cyan")
            log_message(
                "INFO",
                f"   python train_clip.py --model-config {args.model_config} --data-config {args.data_config}",
                color="white",
            )

        except Exception as e:
            log_message("ERROR", "\nDRY RUN FAILED - Configuration error detected!", color="red")
            log_message("ERROR", f"Error: {str(e)}", color="red")
            log_message("INFO", "\nPlease check your configuration files:", color="yellow")
            log_message("INFO", f"   - Model config: {args.model_config}", color="white")
            log_message("INFO", f"   - Data config: {args.data_config}", color="white")

        exit()

    # Normal training mode (original logic)
    # Extract experiment folder name from model config path
    config_path_parts = args.model_config.split("/")
    exp_folder = None
    for i, part in enumerate(config_path_parts):
        if part.startswith("experiments-"):
            exp_folder = part
            break

    if exp_folder is None:
        exp_folder = "experiments"

    if args.version is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.version = f"{timestamp}_{model_name}"  # Put timestamp first for better sorting

    # Create log directory structure: dataset_experiment-folder/model_version
    exp_category = f"{data_config_name}_{exp_folder}"
    log_dir = os.path.join(args.log_dir, exp_category, args.version)
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    config_dir = os.path.join(log_dir, "config")  # Config files now in log directory

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Save config files for reference
    if args.data_config is not None and args.model_config is not None:
        shutil.copy(args.data_config, os.path.join(config_dir, "data_config.yaml"))
        shutil.copy(args.model_config, os.path.join(config_dir, "model_config.yaml"))
    else:
        raise ValueError("Data-config and model-config must be provided")

    # Instantiate data module
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.batch_size = data_cfg.params.cfg.batch_size // args.world_size  # for each gpu
    data = instantiate_from_config(data_cfg)
    data.setup("fit")

    # Set random seed
    set_seed(seed=args.seed, n_gpu=args.world_size)

    # Print core configuration parameters (only on rank 0)
    if args.rank == 0:
        print_core_config(args, log_dir, cfg)

    # Instantiate model
    model_cfg = cfg.model_cfg
    # Inject distributed training parameters into model config
    if model_cfg.params is None:
        model_cfg.params = {}
    model_cfg.params["rank"] = args.rank
    model_cfg.params["world_size"] = args.world_size
    model = instantiate_from_config(model_cfg)

    # Load pretrained weights (weights only, no optimizer/epoch state)
    if args.pretrained_checkpoint:
        log_message("PRETRAINED", f"Loading weights from {args.pretrained_checkpoint}", color="cyan")
        ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded_keys = len(state_dict) - len(unexpected)
        log_message("PRETRAINED", f"Loaded {loaded_keys}/{len(state_dict)} keys", color="green")
        if missing:
            log_message(
                "PRETRAINED",
                f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}",
                color="yellow",
            )
        if unexpected:
            log_message(
                "PRETRAINED",
                f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}",
                color="yellow",
            )

    # End previous SwanLab experiment (if exists)
    try:
        swanlab.finish()
    except:
        pass

    logger = SwanLabLogger(name=args.version, project="mmExpert-CLIP")
    if not args.resume_checkpoint and args.rank == 0:
        for log_file in glob.glob(os.path.join(checkpoints_dir, "*.ckpt")):
            os.remove(log_file)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor="valid/loss_clip",
        filename="epoch_{epoch:02d}_val_{valid/loss_clip:.4f}",
        save_top_k=args.save_top_k,
        mode="min",
        auto_insert_metric_name=False,
        save_last=True,
        save_weights_only=False,
    )

    import time

    class SimpleLogCallback(Callback):
        def __init__(self):
            self._epoch_start = None
        def on_train_epoch_start(self, trainer, pl_module):
            self._epoch_start = time.time()
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.sanity_checking:
                return
            total = trainer.num_training_batches
            elapsed = time.time() - (self._epoch_start or time.time())
            it_per_sec = (batch_idx + 1) / max(elapsed, 0.1)
            remaining_batches = total - (batch_idx + 1)
            eta = remaining_batches / max(it_per_sec, 0.001)
            loss = outputs.get("loss") if isinstance(outputs, dict) else (outputs.item() if hasattr(outputs, "item") else float(outputs))
            print(f"train epoch={trainer.current_epoch} [{batch_idx+1}/{total}] step={trainer.global_step} "
                  f"loss={loss:.4f} {it_per_sec:.2f}it/s eta={eta:.0f}s", flush=True)
        def on_validation_epoch_end(self, trainer, pl_module):
            parts = []
            for k, v in trainer.callback_metrics.items():
                if "loss" in k:
                    val = v.item() if hasattr(v, "item") else float(v)
                    parts.append(f"{k}={val:.4f}")
            print(f"val epoch={trainer.current_epoch} {' '.join(parts)}", flush=True)


    trainer = Trainer(
        accelerator="gpu",
        devices=args.world_size,
        strategy=args.strategy,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=cfg.model_cfg.params.max_epochs,
        num_sanity_val_steps=2,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[checkpoint_callback, SimpleLogCallback()],
        enable_progress_bar=False,
    )

    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=[data.val_dataloader()],
        ckpt_path=args.resume_checkpoint if args.resume_checkpoint else None,
    )
