"""Debug script to test checkpoint on training dataset - mimics training flow"""

import argparse
import os
os.chdir('/root/autodl-tmp/mmExpert')

import sys
sys.path.append('.')

import random
import numpy as np

import torch
from tqdm import tqdm
from termcolor import colored

from src.llm.datamodule import WaveLLMDataModule
from src.llm.utils.common_utils import load_model_from_checkpoint
from src.llm.utils.config_loader import load_yaml_config


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch, device):
    """Move nested batch to target device."""
    batch_gpu = {}
    for key, value in batch.items():
        if key == 'mmwave_features' and isinstance(value, dict):
            batch_gpu[key] = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in value.items()
            }
        elif torch.is_tensor(value):
            batch_gpu[key] = value.to(device)
        else:
            batch_gpu[key] = value
    return batch_gpu


def print_model_diagnostics(model):
    """Print model diagnostic information"""
    print("")
    print(colored("[DIAGNOSTIC]", "cyan") + " Checking model state...")
    print(f"  Model is in {'train' if model.training else 'eval'} mode")
    print(f"  Model.model is in {'train' if model.model.training else 'eval'} mode")
    
    # Check LoRA status
    if hasattr(model.model, 'peft_config'):
        print(colored("[INFO]", "green") + " PEFT/LoRA is active")
        peft_type = model.model.peft_config.get('default', {})
        if hasattr(peft_type, 'r'):
            print(f"  LoRA rank: {peft_type.r}")
            print(f"  LoRA alpha: {peft_type.lora_alpha}")
            print(f"  LoRA dropout: {peft_type.lora_dropout}")
    else:
        print(colored("[WARNING]", "red") + " PEFT/LoRA is NOT active!")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Debug checkpoint on training dataset")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--max_batches", type=int, default=10,
                        help="Maximum number of batches to test")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for testing")
    parser.add_argument("--use_train_mode", action='store_true',
                        help="Use train mode (with dropout) instead of eval mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    set_global_seed(args.seed)

    # Load model using common_utils approach
    model, _ = load_model_from_checkpoint(
        args.model_checkpoint,
        args.config,
        args.data_root
    )
    
    # Set model mode
    if args.use_train_mode:
        print(colored("[INFO]", "yellow") + " Using TRAIN mode (with dropout, matching training)")
        model.train()
        model.model.train()
    else:
        print(colored("[INFO]", "green") + " Using EVAL mode (no dropout)")
        # Already in eval mode from load_model_for_debug
        
    model.debug_mode = True
    
    # Print diagnostics AFTER setting mode
    print_model_diagnostics(model)
    
    # Setup data module (same as training)
    print(colored("[LOAD]", "cyan") + f" Loading training dataset from {args.data_root}")
    # Build dataloader (matching training)
    cfg_yaml = load_yaml_config(args.config)
    cfg_yaml.data_cfg.data_root = args.data_root
    cfg_yaml.model_cfg.encoder_path = args.data_root
    cfg_yaml.data_cfg.batch_size = args.batch_size

    data_module = WaveLLMDataModule(cfg_yaml.data_cfg)
    data_module.setup(stage='fit')
    dataloader = data_module.val_dataloader()
    
    # Test on training data with bf16-mixed (same as training)
    print(colored("[INFO]", "green") + " Using bf16-mixed precision (matching training)")
    print(colored("[PROGRESS]", "yellow") + f" Running {args.max_batches} batches")
    print("")
    
    all_losses = []
    
    device = next(model.parameters()).device
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for step, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=args.max_batches)):
                if step >= args.max_batches:
                    break

                batch_gpu = move_batch_to_device(batch, device)

                outputs = model(batch_gpu)
                loss_val = outputs['loss'].item()
                all_losses.append(loss_val)

                tqdm.write(
                    colored("[BATCH]", "yellow")
                    + f" {step+1}/{args.max_batches} - Loss: {loss_val:.6f}"
                )
    
    # Print summary
    print("")
    print(colored("[RESULTS]", "blue") + " " + "="*70)
    print(colored("[RESULTS]", "blue") + " TRAINING LOSS ANALYSIS")
    print(colored("[RESULTS]", "blue") + " " + "="*70)
    print("")
    print(f"Total steps: {len(all_losses)}")
    avg_loss = sum(all_losses) / len(all_losses)
    print(f"Average loss: {avg_loss:.6f}")
    print(f"Min loss: {min(all_losses):.6f}")
    print(f"Max loss: {max(all_losses):.6f}")
    print(f"Loss std: {torch.tensor(all_losses).std().item():.6f}")
    print("")
    
    # Print all losses for detailed inspection
    print(colored("[DETAILS]", "cyan") + " Per-step losses:")
    for i, loss_val in enumerate(all_losses):
        print(f"  Step {i+1}: {loss_val:.6f}")
    
    print("")
    print(colored("[SUCCESS]", "green") + " Debug test completed!")
    
    # Analysis
    avg_loss = sum(all_losses) / len(all_losses)
    if avg_loss < 0.5:
        print(colored("[ANALYSIS]", "green") + " ✓ Loss is low - model fits training data well")
    elif avg_loss < 2.0:
        print(colored("[ANALYSIS]", "yellow") + " ⚠ Loss is moderate - model partially fits training data")
    else:
        print(colored("[ANALYSIS]", "red") + " ✗ Loss is high - model may not have trained properly")
    
    # Suggestions
    print("")
    print(colored("[SUGGESTIONS]", "cyan") + " If loss is unexpectedly high:")
    print("  1. Try --use_train_mode to match training's dropout behavior")
    print("  2. Check if checkpoint is from the correct training run")
    print("  3. Verify that training actually converged (check SwanLab logs)")
    print("  4. Compare with validation loss during training")


if __name__ == "__main__":
    main()
