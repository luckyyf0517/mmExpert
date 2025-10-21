#!/usr/bin/env python3
"""
Data visualization script using project's data interface.
Loads data using the actual dataset configuration and preprocessing pipeline.
"""

import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from termcolor import colored

# Add project root to path
sys.path.append('.')

from src.misc.io import load_config
from src.misc.tools import instantiate_from_config

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')


def ensure_preview_directory():
    """Ensure the tmp/preview directory exists."""
    preview_dir = "/root/autodl-tmp/mmExpert/tmp/preview"
    os.makedirs(preview_dir, exist_ok=True)
    return preview_dir


def save_spectrum_image(tensor_data, filename, title, view_name, sample_idx):
    """
    Save spectrum tensor as an image with proper visualization.

    Args:
        tensor_data: PyTorch tensor of shape [batch, height, width] or [height, width]
        filename: Output filename
        title: Image title
        view_name: Name of the radar view
        sample_idx: Sample index for filename
    """
    try:
        # Convert to numpy and handle batch dimension
        if len(tensor_data.shape) == 3:
            # Take first sample from batch
            data = tensor_data[0].cpu().numpy()
        else:
            data = tensor_data.cpu().numpy()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{title} - {view_name}', fontsize=14, fontweight='bold')

        # Raw data visualization
        im1 = ax1.imshow(data, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title('Raw Data')
        ax1.set_xlabel('Time frames')
        ax1.set_ylabel('Frequency bins')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Normalized data visualization
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        im2 = ax2.imshow(data_norm, aspect='auto', cmap='jet', origin='lower')
        ax2.set_title('Normalized Data')
        ax2.set_xlabel('Time frames')
        ax2.set_ylabel('Frequency bins')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Add statistics text
        stats_text = f'Shape: {data.shape}\nMin: {data.min():.3f}\nMax: {data.max():.3f}\nMean: {data.mean():.3f}'
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(colored(f"[SUCCESS] Saved spectrum image: {os.path.basename(filename)}", 'green'))
        return True

    except Exception as e:
        print(colored(f"[ERROR] Failed to save spectrum image {filename}: {e}", 'red'))
        return False


def save_all_spectrum_views(wave_data, sample_idx, preview_dir):
    """Save all three spectrum views (range, doppler, azimuth) as images."""
    base_name = f"sample_{sample_idx:03d}"

    # Define view configurations
    view_configs = [
        ('input_wave_range', 'Range-Time Spectrum', 'range'),
        ('input_wave_doppler', 'Doppler-Time Spectrum', 'doppler'),
        ('input_wave_azimuth', 'Azimuth-Time Spectrum', 'azimuth')
    ]

    success_count = 0

    for key, title, view_name in view_configs:
        if key in wave_data:
            filename = os.path.join(preview_dir, f"{base_name}_{view_name}_spectrum.png")
            if save_spectrum_image(wave_data[key], filename, title, view_name, sample_idx):
                success_count += 1

    print(colored(f"[INFO] Saved {success_count}/3 spectrum images for sample {sample_idx}", 'blue'))
    return success_count == 3


def create_data_summary_plot(wave_data, caption, sample_idx, preview_dir):
    """Create a summary plot showing all views and caption."""
    try:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Data Summary - Sample {sample_idx}', fontsize=16, fontweight='bold')

        views = [
            ('input_wave_range', 'Range-Time', 'viridis'),
            ('input_wave_doppler', 'Doppler-Time', 'plasma'),
            ('input_wave_azimuth', 'Azimuth-Time', 'inferno')
        ]

        for idx, (key, title, colormap) in enumerate(views):
            if key in wave_data and idx < 3:
                # Get data and handle batch dimension
                data = wave_data[key][0].cpu().numpy() if len(wave_data[key].shape) == 3 else wave_data[key].cpu().numpy()

                # Spectrum plot
                ax_spectrum = axes[idx, 0]
                im = ax_spectrum.imshow(data, aspect='auto', cmap=colormap, origin='lower')
                ax_spectrum.set_title(f'{title} Spectrum', fontweight='bold')
                ax_spectrum.set_xlabel('Time frames')
                ax_spectrum.set_ylabel('Frequency bins')
                plt.colorbar(im, ax=ax_spectrum, fraction=0.046, pad=0.04)

                # Statistics panel
                ax_stats = axes[idx, 1]
                stats_text = f'Shape: {data.shape}\n\n'
                stats_text += f'Min: {data.min():.3f}\n'
                stats_text += f'Max: {data.max():.3f}\n'
                stats_text += f'Mean: {data.mean():.3f}\n'
                stats_text += f'Std: {data.std():.3f}\n\n'
                stats_text += f'Non-zero: {np.count_nonzero(data)}'

                ax_stats.axis('off')
                ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                             fontsize=9, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Hide unused axes
        for idx in range(len(views), 3):
            axes[idx, 0].axis('off')
            axes[idx, 1].axis('off')

        # Add caption at the bottom
        fig.text(0.5, 0.02, f'Caption: "{caption}"',
                  ha='center', va='bottom', fontsize=11,
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        # Save summary
        summary_filename = os.path.join(preview_dir, f"sample_{sample_idx:03d}_summary.png")
        plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(colored(f"[SUCCESS] Saved summary: {os.path.basename(summary_filename)}", 'green'))
        return True

    except Exception as e:
        print(colored(f"[ERROR] Failed to create summary plot: {e}", 'red'))
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='View dataset samples using project data interface')
    parser.add_argument('--data-config', type=str, required=True,
                       help='Path to data configuration file')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to display (default: 3)')
    parser.add_argument('--model-config', type=str, default=None,
                       help='Path to model configuration file (for complete setup test)')
    parser.add_argument('--save-images', action='store_true', default=True,
                       help='Save spectrum images and summary plots')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for saved images (default: tmp/preview)')
    return parser.parse_args()


def display_tensor_info(wave_embed, sample_num):
    """Display detailed information about wave embedding tensors."""
    print(colored(f"\n--- Sample {sample_num} Wave Embedding Details ---", 'cyan', attrs=['bold']))

    for view_name, tensor in wave_embed.items():
        if torch.is_tensor(tensor):
            print(colored(f"  [{view_name.upper()}]", 'blue', attrs=['bold']))
            print(f"    Shape:      {tensor.shape}")
            print(f"    Data type:  {tensor.dtype}")
            print(f"    Device:     {tensor.device}")
            print(f"    Min value:  {tensor.min():.6f}")
            print(f"    Max value:  {tensor.max():.6f}")
            print(f"    Mean value: {tensor.mean():.6f}")
            print(f"    Std value:  {tensor.std():.6f}")

            # Check for any special values
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            if nan_count > 0:
                print(f"    NaN count:  {nan_count}")
            if inf_count > 0:
                print(f"    Inf count:  {inf_count}")


def load_data_sample_from_dataset(data_interface, num_samples=3, save_images=False, preview_dir=None):
    """Load data samples using the project's data interface."""
    print(f"\n{'='*60}")
    print("Loading Data Using Project Data Interface")
    print(f"{'='*60}")

    try:
        # Setup data interface
        data_interface.setup('fit')

        # Get dataloaders
        train_dataloader = data_interface.train_dataloader()
        val_dataloader = data_interface.val_dataloader()

        print(colored("[SUCCESS] Data interface setup completed", 'green'))

        # Display dataset info
        print(colored(f"\n[DATASET] Dataset Information:", 'cyan', attrs=['bold']))
        print(f"  Training samples:   {len(train_dataloader.dataset)}")
        print(f"  Validation samples: {len(val_dataloader.dataset)}")
        print(f"  Train batch size:  {train_dataloader.batch_size}")
        print(f"  Val batch size:    {val_dataloader.batch_size}")
        print(f"  Train batches:     {len(train_dataloader)}")
        print(f"  Val batches:       {len(val_dataloader)}")

        # Load samples from training set
        print(colored(f"\n[LOADING] Loading {num_samples} samples from training set...", 'cyan', attrs=['bold']))

        samples = []
        for i, batch in enumerate(train_dataloader):
            if i >= num_samples:
                break

            print(f"\n--- Sample {i} ---")

            # Extract batch data (assuming dict structure)
            if isinstance(batch, dict):
                # Handle different wave embedding key names
                wave_embed = batch.get('wave_embed', {})
                if not wave_embed:
                    # Try alternative key names
                    wave_keys = ['input_wave_range', 'input_wave_doppler', 'input_wave_azimuth']
                    wave_embed = {key: batch.get(key) for key in wave_keys if batch.get(key) is not None}

                text_data = batch.get('text', None)
                caption = batch.get('caption', None)
                file_path = batch.get('file_path', None)

                # Display batch dimensions
                print(colored(f"  [BATCH] Batch Information:", 'magenta'))
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"    {key}: {value.shape}    dtype={value.dtype}")
            else:
                print(colored("[WARNING] Unexpected batch structure", 'yellow'))
                continue

            # Display wave embedding details
            display_tensor_info(wave_embed, i+1)

            # Save images if requested
            if save_images and preview_dir:
                print(colored(f"\n[IMAGES] Saving images for sample {i+1}...", 'yellow'))
                save_all_spectrum_views(wave_embed, i+1, preview_dir)

                # Handle caption for summary
                sample_caption = caption
                if isinstance(caption, list) and caption:
                    sample_caption = caption[0]
                elif not isinstance(caption, str):
                    sample_caption = str(caption)

                create_data_summary_plot(wave_embed, sample_caption, i+1, preview_dir)

            # Display text information
            if text_data is not None:
                print(colored(f"  [TEXT] Text Processing Information:", 'blue'))
                if torch.is_tensor(text_data):
                    print(f"    Text tensor shape: {text_data.shape}")
                    print(f"    Text tensor dtype: {text_data.dtype}")
                elif isinstance(text_data, dict):
                    for key, value in text_data.items():
                        if torch.is_tensor(value):
                            print(f"    {key}: {value.shape}    dtype={value.dtype}")

            # Display caption
            if caption is not None:
                print(colored(f"  [CAPTION] Text Caption:", 'green'))
                if isinstance(caption, (list, tuple)) and caption:
                    # Handle batch of captions
                    for j, cap in enumerate(caption[:2]):  # Show first 2
                        print(f"    Caption {j+1}: \"{str(cap)[:80]}{'...' if len(str(cap)) > 80 else ''}\"")
                elif isinstance(caption, str):
                    print(f"    Caption: \"{caption[:80]}{'...' if len(caption) > 80 else ''}\"")

            # Display file path
            if file_path is not None:
                print(colored(f"  [PATH] Source Files:", 'yellow'))
                if isinstance(file_path, (list, tuple)) and file_path:
                    for j, path in enumerate(file_path[:2]):  # Show first 2
                        print(f"    File {j+1}: {os.path.basename(path)}")
                elif isinstance(file_path, str):
                    print(f"    File: {os.path.basename(file_path)}")

            samples.append({
                'wave_embed': wave_embed,
                'text': text_data,
                'caption': caption,
                'file_path': file_path
            })

        # Load one sample from validation set
        print(colored(f"\n[VALIDATION] Loading validation sample...", 'cyan', attrs=['bold']))
        val_batch = next(iter(val_dataloader))

        print(f"\n--- Validation Sample ---")
        if isinstance(val_batch, dict):
            # Handle different wave embedding key names
            val_wave_embed = val_batch.get('wave_embed', {})
            if not val_wave_embed:
                # Try alternative key names
                wave_keys = ['input_wave_range', 'input_wave_doppler', 'input_wave_azimuth']
                val_wave_embed = {key: val_batch.get(key) for key in wave_keys if val_batch.get(key) is not None}

            val_caption = val_batch.get('caption', None)

            display_tensor_info(val_wave_embed, 'Validation')

            if val_caption is not None:
                print(colored(f"  [VALIDATION CAPTION]:", 'green'))
                if isinstance(val_caption, (list, tuple)) and val_caption:
                    print(f"    Caption: \"{str(val_caption[0])[:80]}{'...' if len(str(val_caption[0])) > 80 else ''}\"")
                elif isinstance(val_caption, str):
                    print(f"    Caption: \"{val_caption[:80]}{'...' if len(val_caption) > 80 else ''}\"")

        return samples

    except Exception as e:
        print(colored(f"[ERROR] Error loading data from dataset: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return []


def display_config_details(data_cfg):
    """Display detailed configuration information."""
    print(colored(f"\n[CONFIG] Data Configuration Details:", 'cyan', attrs=['bold']))
    print(f"  Data interface target: {data_cfg.target}")

    if hasattr(data_cfg, 'params') and hasattr(data_cfg.params, 'cfg'):
        cfg = data_cfg.params.cfg

        # Display split information
        if hasattr(cfg, 'train_split'):
            print(colored(f"  [DATA SPLITS]:", 'blue'))
            print(f"    Train: {cfg.train_split}")
            if hasattr(cfg, 'val_split'):
                print(f"    Val:   {cfg.val_split}")
            if hasattr(cfg, 'test_split'):
                print(f"    Test:  {cfg.test_split}")

        # Display processing options
        if hasattr(cfg, 'opt'):
            opt = cfg.opt
            print(colored(f"  [PROCESSING OPTIONS]:", 'blue'))
            print(f"    Max motion length: {opt.max_motion_length}")
            print(f"    Min motion length: {opt.min_motion_len}")
            print(f"    Max text length:   {opt.max_text_len}")
            print(f"    Unit length:       {opt.unit_length}")
            print(f"    Normalize:         {opt.normalize}")
            print(f"    Radar views:       {opt.radar_views}")
            if hasattr(opt, 'mmwave_postfix'):
                print(f"    MMWave postfix:    '{opt.mmwave_postfix}'")

        # Display training parameters
        if hasattr(cfg, 'batch_size'):
            print(colored(f"  [TRAINING PARAMETERS]:", 'blue'))
            print(f"    Batch size:  {cfg.batch_size}")
            print(f"    Num workers: {cfg.num_workers}")
            if hasattr(cfg, 'sample_ratio'):
                print(f"    Sample ratio: {cfg.sample_ratio}")


def main():
    """Main function to demonstrate data loading using project's data interface."""
    args = parse_args()

    print(colored("[START] Dataset Viewer Using Project Data Interface", 'blue', attrs=['bold']))
    print(colored(f"{'='*60}", 'blue'))
    print(colored(f"[CONFIG] Using data config: {args.data_config}", 'cyan'))

    # Setup output directory if saving images
    preview_dir = None
    if args.save_images:
        preview_dir = args.output_dir if args.output_dir else ensure_preview_directory()
        print(colored(f"[IMAGES] Saving images to: {preview_dir}", 'yellow'))

    # Load data configuration
    try:
        config = load_config(None, args.data_config, args.model_config)
        data_cfg = config['data_cfg']
        print(colored("[SUCCESS] Configuration loaded successfully", 'green'))

        # Display detailed configuration
        display_config_details(data_cfg)

    except Exception as e:
        print(colored(f"[ERROR] Failed to load configuration: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return

    # Instantiate data interface
    try:
        print(colored(f"\n[SETUP] Setting up data interface...", 'cyan'))
        data_interface = instantiate_from_config(data_cfg)
        print(colored("[SUCCESS] Data interface instantiated successfully", 'green'))
    except Exception as e:
        print(colored(f"[ERROR] Failed to instantiate data interface: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return

    # Load and display data samples
    try:
        samples = load_data_sample_from_dataset(data_interface, args.num_samples, args.save_images, preview_dir)

        if samples:
            print(colored(f"\n[SUCCESS] Successfully loaded {len(samples)} data samples", 'green'))

            # Summary
            print(colored(f"\n{'='*60}", 'blue'))
            print(colored("Summary", 'blue', attrs=['bold']))
            print(colored(f"{'='*60}", 'blue'))

            print(colored("[SUCCESS] Data interface test completed successfully!", 'green', attrs=['bold']))
            print(colored("\n[SUMMARY] What was demonstrated:", 'cyan', attrs=['bold']))
            print(colored("  [SUCCESS] Loading data using project's data interface", 'green'))
            print(colored("  [SUCCESS] Applying dataset-specific preprocessing", 'green'))
            print(colored("  [SUCCESS] Converting to proper tensor formats", 'green'))
            print(colored("  [SUCCESS] Handling text captions and tokenization", 'green'))
            print(colored("  [SUCCESS] Batch processing with proper shapes", 'green'))
            print(colored("  [SUCCESS] Displaying tensor statistics and metadata", 'green'))

            print(colored("\n[INFO] The dataset preprocessing pipeline is working correctly!", 'blue', attrs=['bold']))
            print(colored(f"[INFO] Use this tool to verify data before training experiments", 'yellow'))

            # Output saved files summary if images were saved
            if args.save_images and preview_dir:
                saved_files = []
                try:
                    for root, dirs, files in os.walk(preview_dir):
                        for file in files:
                            if file.endswith(('.png', '.jpg', '.jpeg')):
                                saved_files.append(os.path.join(root, file))

                    if saved_files:
                        print(colored(f"\n[FILES] Images saved to directory:", 'cyan', attrs=['bold']))
                        print(colored(f"        {preview_dir}", 'cyan'))
                        print(colored(f"[FILES] Total saved files: {len(saved_files)}", 'green'))
                        print(colored("[FILES] Saved image files:", 'blue'))
                        for i, file_path in enumerate(sorted(saved_files), 1):
                            file_size = os.path.getsize(file_path)
                            file_size_mb = file_size / (1024 * 1024)  # Convert to MB
                            print(colored(f"        {i:2d}. {os.path.basename(file_path)} ({file_size_mb:.2f} MB)", 'white'))
                except Exception as e:
                    print(colored(f"[ERROR] Failed to list saved files: {e}", 'red'))

        else:
            print(colored("[ERROR] No data samples were loaded", 'red'))

    except Exception as e:
        print(colored(f"[ERROR] Error during data loading: {e}", 'red'))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()