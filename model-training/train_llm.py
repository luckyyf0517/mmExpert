"""Training entry point for fine-tuning Phi3 with mmwave features using PyTorch Lightning"""

import argparse
import os
import warnings
from datetime import datetime
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import LearningRateMonitor

# Ignore PyTorch Lightning model summary precision warning (bf16-mixed not supported for size estimation)
warnings.filterwarnings('ignore', message='.*Precision bf16-mixed is not supported by the model summary.*')

from src.llm.utils.config_loader import load_yaml_config
from src.llm.utils.deepspeed_utils import add_deepspeed_args, get_train_ds_config
from src.llm.utils.common_utils import (
    override_config,
    save_training_artifacts,
    initialize_trainer_from_checkpoint,
)
from src.logger import log_message
from src.llm.datamodule import WaveLLMDataModule
from src.llm.trainer import WaveLLMTrainer

import sys
if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune Phi3 with mmwave features")

    # Config
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Optional WaveLLM checkpoint directory to initialize from before continuing training')

    # Data root path (overrides config file)
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to feature directory (overrides config file)')
    parser.add_argument('--train_split', type=str, default=None,
                        help='Dataset train split to use for training (overrides config file)')
    parser.add_argument('--val_split', type=str, default=None,
                        help='Validation split used during training (overrides config file)')
    parser.add_argument('--static_bg_npz', type=str, default=None,
                        help='Path to static background three-view NPZ for on-the-fly overlay training')
    parser.add_argument('--range_drop_prob', type=float, default=0.0,
                        help='Probability to zero out range_time view (default 0)')
    parser.add_argument('--doppler_drop_prob', type=float, default=0.0,
                        help='Probability to zero out doppler_time view (default 0)')
    parser.add_argument('--azimuth_drop_prob', type=float, default=0.0,
                        help='Probability to zero out azimuth_time view (default 0)')

    # Training parameters (override config file)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config file)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers (overrides config file)')
    parser.add_argument('--use_random_question_for_caption', type=lambda x: x.lower() == 'true', default=None,
                        help='Whether to use random question selection (true/false, overrides config file)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate for training (overrides config file)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay for training (overrides config file)')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Warmup steps for training (overrides config file)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum number of training epochs (overrides config file)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum number of training steps (overrides max_epochs)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Gradient accumulation steps (overrides config file)')

    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=None,
                        help='LoRA rank parameter (overrides config file)')
    parser.add_argument('--lora_alpha', type=int, default=None,
                        help='LoRA alpha parameter (overrides config file)')

    # Basic training args
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
        cfg.data_cfg.data_root = args.data_root
        cfg.model_cfg.encoder_path = args.data_root
        log_message("CONFIG", f"Using data_root from command line: {args.data_root}", color="blue")
    else:
        if not cfg.data_cfg.data_root:
            raise ValueError("data_root must be provided either via --data_root argument or in config file")
        if not cfg.model_cfg.encoder_path:
            cfg.model_cfg.encoder_path = cfg.data_cfg.data_root

    # Override training parameters if provided via command line
    if args.train_split is not None:
        # Parse comma-separated split files into list
        if ',' in args.train_split:
            cfg.data_cfg.train_split = args.train_split.split(',')
            log_message("CONFIG", f"Using train_split from command line: {cfg.data_cfg.train_split}", color="blue")
        else:
            cfg.data_cfg.train_split = args.train_split
            log_message("CONFIG", f"Using train_split from command line: {args.train_split}", color="blue")

    if args.val_split is not None:
        # Parse comma-separated split files into list
        if ',' in args.val_split:
            cfg.data_cfg.val_split = args.val_split.split(',')
            log_message("CONFIG", f"Using val_split from command line: {cfg.data_cfg.val_split}", color="blue")
        else:
            cfg.data_cfg.val_split = args.val_split
            log_message("CONFIG", f"Using val_split from command line: {args.val_split}", color="blue")
    override_config(cfg.data_cfg, 'batch_size', args.batch_size, 'batch_size')
    override_config(cfg.data_cfg, 'num_workers', args.num_workers, 'num_workers')
    override_config(cfg.data_cfg, 'use_random_question_for_caption', args.use_random_question_for_caption, 'use_random_question_for_caption')
    if args.static_bg_npz is not None:
        cfg.data_cfg.static_bg_npz = args.static_bg_npz
        log_message("CONFIG", f"Using static_bg_npz from command line: {args.static_bg_npz}", color="blue")
    for prob_name in ('range_drop_prob', 'doppler_drop_prob', 'azimuth_drop_prob'):
        prob_val = getattr(args, prob_name, 0.0)
        if prob_val > 0:
            cfg.data_cfg[prob_name] = prob_val
            log_message("CONFIG", f"Using {prob_name} from command line: {prob_val}", color="blue")
    override_config(cfg.training, 'max_epochs', args.max_epochs, 'max_epochs')
    override_config(cfg.training, 'gradient_accumulation_steps', args.gradient_accumulation_steps, 'gradient_accumulation_steps')
    override_config(cfg.training, 'learning_rate', args.learning_rate, 'learning_rate')
    override_config(cfg.training, 'weight_decay', args.weight_decay, 'weight_decay')
    override_config(cfg.training, 'warmup_steps', args.warmup_steps, 'warmup_steps')

    # Override LoRA parameters if provided
    override_config(cfg.model_cfg.peft_config, 'r', args.lora_r, 'lora_r')
    override_config(cfg.model_cfg.peft_config, 'lora_alpha', args.lora_alpha, 'lora_alpha')

    # Set seeds
    pl.seed_everything(args.seed)

    # Add timestamp prefix to log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_parent = os.path.dirname(cfg.log_dir)
    log_dir_name = os.path.basename(cfg.log_dir)
    cfg.log_dir = os.path.join(log_dir_parent, f"{timestamp}_{log_dir_name}")
    log_message("CONFIG", f"Using log directory: {cfg.log_dir}", color="blue")

    # Initialize data module
    data_module = WaveLLMDataModule(cfg.data_cfg)

    # Create model (LightningModule)
    # Pass model_cfg with training config added
    model_cfg_with_training = deepcopy(cfg.model_cfg)
    model_cfg_with_training.training = cfg.training
    model_cfg_with_training.data_cfg = cfg.data_cfg

    model = WaveLLMTrainer(model_cfg_with_training)
    if args.init_checkpoint is not None:
        initialize_trainer_from_checkpoint(model, args.init_checkpoint)
        log_message("CONFIG", f"Initialized model from checkpoint: {args.init_checkpoint}", color="blue")

    class EpochCheckpointCallback(pl.Callback):
        """Save adapter and non-LoRA weights after each epoch"""

        def __init__(self, save_dir):
            super().__init__()
            self.save_dir = save_dir

        def on_train_epoch_end(self, trainer, pl_module):
            """Save model artifacts at the end of each training epoch"""
            import os
            from src.llm.utils.common_utils import save_training_artifacts
            from src.llm.trainer import _is_rank_0

            if not _is_rank_0():
                return

            # Create epoch-specific directory
            epoch_dir = os.path.join(self.save_dir, "epochs", f"epoch_{trainer.current_epoch:02d}")
            os.makedirs(epoch_dir, exist_ok=True)

            # Save training artifacts for this epoch (without logging)
            save_training_artifacts(pl_module, epoch_dir, cfg, verbose=False)

    import time

    class SimpleLogCallback(pl.Callback):
        def __init__(self):
            self._epoch_start = None
        def on_train_epoch_start(self, trainer, pl_module):
            self._epoch_start = time.time()
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.sanity_checking:
                return
            total = trainer.num_training_batches
            elapsed = max(time.time() - (self._epoch_start or time.time()), 0.1)
            it_per_sec = (batch_idx + 1) / elapsed
            eta = (total - (batch_idx + 1)) / max(it_per_sec, 0.001)
            loss = outputs.get("loss") if isinstance(outputs, dict) else float(outputs)
            print(f"train epoch={trainer.current_epoch} [{batch_idx+1}/{total}] loss={loss:.4f} {it_per_sec:.2f}it/s eta={eta:.0f}s", flush=True)
        def on_validation_epoch_end(self, trainer, pl_module):
            parts = []
            for k, v in trainer.callback_metrics.items():
                if "loss" in k:
                    val = v.item() if hasattr(v, "item") else float(v)
                    parts.append(f"{k}={val:.4f}")
            if parts:
                print(f"val epoch={trainer.current_epoch} {' '.join(parts)}", flush=True)

    # Setup callbacks - no checkpoint saving (we'll save LoRA adapter manually)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EpochCheckpointCallback(save_dir=cfg.log_dir),
        SimpleLogCallback()
    ]

    # Setup SwanLab logger
    swanlab_logger = None  # SwanLabLogger disabled

    # Create strategy based on --no_deepspeed flag
    if args.no_deepspeed:
        # Use native PyTorch Lightning for single GPU (faster, simpler)
        strategy = "auto"
        log_message("CONFIG", "Using native PyTorch Lightning (DeepSpeed disabled)", color="blue")
    else:
        # Use DeepSpeed strategy
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
        'enable_progress_bar': False,
        'deterministic': True,
        'callbacks': callbacks,
        'logger': swanlab_logger,
        'default_root_dir': cfg.log_dir
    }

    # Use max_steps if provided, otherwise use max_epochs
    if args.max_steps is not None:
        trainer_kwargs['max_steps'] = args.max_steps
        log_message("CONFIG", f"Using max_steps: {args.max_steps}", color="blue")
    else:
        trainer_kwargs['max_epochs'] = cfg.training.max_epochs

    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    log_message("PROGRESS", "Starting training...", color="yellow")
    trainer.fit(model, datamodule=data_module)
    log_message("SUCCESS", "Training completed successfully", color="green")

    # Save final training artifacts to current path (only on rank 0)
    from src.llm.trainer import _is_rank_0
    if _is_rank_0():
        # Also save to log_dir for compatibility
        log_message("INFO", "Rank 0: Saving training artifacts to log_dir", color="green")
        save_training_artifacts(model, cfg.log_dir, cfg)
    else:
        log_message("INFO", "Non-rank 0: Skipping save", color="yellow")


if __name__ == "__main__":
    main()
