"""DeepSpeed utilities for LLM training"""

import argparse
import os


def get_train_ds_config(args):
    """Generate DeepSpeed training configuration

    Args:
        args: Command line arguments with DeepSpeed settings

    Returns:
        dict: DeepSpeed configuration dictionary
    """
    # Get world_size from environment if available (set by deepspeed launcher), otherwise use args
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    
    micro_batch = args.batch_size if args.batch_size is not None else 1
    grad_accum = getattr(args, 'gradient_accumulation_steps', None)
    if grad_accum is None or grad_accum <= 0:
        grad_accum = 1
    
    ds_config = {
        "train_batch_size": micro_batch * grad_accum * world_size,
        "train_micro_batch_size_per_gpu": micro_batch,
        # gradient_accumulation_steps is handled by PyTorch Lightning's accumulate_grad_batches
        "zero_optimization": {
            "stage": args.zero_stage,
            "offload_optimizer": {
                "device": "cpu" if args.offload else "none",
                "pin_memory": True
            } if args.offload else {},
            "offload_param": {
                "device": "cpu" if args.offload else "none",
                "pin_memory": True
            } if args.offload else {},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "fp16": {
            "enabled": args.dtype == "fp16",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15
        },
        "bf16": {
            "enabled": args.dtype == "bf16"
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 1,
        "wall_clock_breakdown": False,
        "monitor_config": {
            "enabled": False,
            "tag": "monitor",
            "csv_monitor": {
                "enabled": True,
                "output_path": "./monitor/"
            },
            "tensorboard": {
                "enabled": True,
                "output_path": "./tensorboard/"
            },
            "wandb": {
                "enabled": False
            }
        }
    }

    return ds_config


def add_deepspeed_args(parser):
    """Add DeepSpeed related arguments to parser

    Args:
        parser (argparse.ArgumentParser): Argument parser

    Returns:
        argparse.ArgumentParser: Updated parser with DeepSpeed arguments
    """
    parser.add_argument('--zero_stage', type=int, default=2,
                        choices=[0, 1, 2, 3],
                        help='ZeRO optimization stage')
    parser.add_argument('--offload', action='store_true',
                        help='Offload optimizer states and parameters to CPU')
    parser.add_argument('--dtype', choices=['fp16', 'bf16', 'fp32'],
                        default='bf16',
                        help='Training data type')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of processes for distributed training')

    return parser