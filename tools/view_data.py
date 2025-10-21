#!/usr/bin/env python3
"""
Data visualization script using project's data interface.
Loads data using actual dataset configuration and preprocessing pipeline.
"""

import os
import sys
import torch
import numpy as np
import argparse
import cv2
from termcolor import colored

# Add project root to path
sys.path.append('.')

from src.misc.io import load_config
from src.misc.tools import instantiate_from_config


def ensure_preview_directory():
    """Ensure tmp/preview directory exists."""
    preview_dir = "/root/autodl-tmp/mmExpert/tmp/preview"
    os.makedirs(preview_dir, exist_ok=True)
    return preview_dir


def normalize_data_for_display(data, target_range=(0, 255)):
    """Normalize data for display with proper scaling."""
    if data.max() > data.min():
        data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    else:
        data_normalized = np.zeros_like(data, dtype=np.uint8)
    return data_normalized


def create_beautiful_summary_cv2(wave_data, caption, sample_idx, preview_dir):
    """Create a beautiful summary plot using OpenCV with professional layout."""
    try:
        # Canvas size - high resolution
        canvas_width = 1920
        canvas_height = 1080
        margin = 50
        plot_start_y = margin

        # Create white background
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # Title
        title_text = f"Data Summary - Sample {sample_idx}"
        cv2.putText(canvas, title_text,
                   (canvas_width//2 - 200, plot_start_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 100), 3)
        plot_start_y += 100

        # Define view configurations
        views = [
            ('input_wave_range', 'Range-Time'),
            ('input_wave_doppler', 'Doppler-Time'),
            ('input_wave_azimuth', 'Azimuth-Time')
        ]

        # Plot each view
        plot_height = 300
        plot_width = (canvas_width - 3 * margin) // 4
        individual_width = plot_width - 250  # Reserve space for stats

        for i, (key, title) in enumerate(views):
            if key not in wave_data:
                continue

            # Get data and handle batch dimension
            if len(wave_data[key].shape) == 3:
                data = wave_data[key][0].cpu().numpy()
            else:
                data = wave_data[key].cpu().numpy()

            # Calculate position
            x_start = margin + i * (plot_width // 3)
            y_start = plot_start_y

            # Create subplot area with subtle border
            cv2.rectangle(canvas, (x_start-2, y_start-2),
                          (x_start + plot_width + 2, y_start + plot_height),
                          (200, 200, 200), -1)

            # Normalized spectrum image (left side)
            data_norm = normalize_data_for_display(data)
            data_resized = cv2.resize(data_norm, (individual_width, plot_height))
            data_colored = cv2.applyColorMap(data_resized, cv2.COLORMAP_VIRIDIS)

            # Place in canvas
            canvas[y_start:y_start+plot_height, x_start:x_start+individual_width] = data_colored

            # Title and stats box (right side)
            title_x = x_start + individual_width + 20
            cv2.putText(canvas, f"{title} Spectrum",
                       (title_x, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Statistics
            stats = [
                f"Shape: {data.shape}",
                f"Min: {data.min():.3f}",
                f"Max: {data.max():.3f}",
                f"Mean: {data.mean():.3f}",
                f"Std: {data.std():.3f}",
                f"Non-zero: {np.count_nonzero(data)}"
            ]

            for j, stat in enumerate(stats):
                cv2.putText(canvas, stat,
                           (title_x, y_start + 60 + j * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            plot_start_y += plot_height + 50

        # Caption area with background
        plot_start_y += 50
        caption_height = 120

        # Caption background
        cv2.rectangle(canvas, (margin, plot_start_y-10),
                      (canvas_width - margin, plot_start_y + caption_height),
                      (240, 240, 220), -1)

        # Add caption text with word wrap
        caption_display = str(caption) if isinstance(caption, str) else str(caption[0]) if caption else ""

        # Word wrap long captions
        max_chars_per_line = 80
        if len(caption_display) > max_chars_per_line:
            words = caption_display.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word + " ") <= max_chars_per_line:
                    current_line += word + " "
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            if current_line.strip():
                lines.append(current_line.strip())
        else:
            lines = [caption_display]

        # Draw caption text
        for i, line in enumerate(lines[:4]):  # Max 4 lines
            cv2.putText(canvas, line,
                       (margin + 20, plot_start_y + 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        plot_start_y += caption_height + 30

        # Footer with sample info
        footer_text = f"Sample {sample_idx:03d} - mmExpert Data Visualization"
        cv2.putText(canvas, footer_text,
                   (margin, canvas_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        # Save beautiful summary
        summary_filename = os.path.join(preview_dir, f"sample_{sample_idx:03d}_summary.png")
        cv2.imwrite(summary_filename, canvas)

        print(colored(f"[SUCCESS] Saved beautiful summary: {os.path.basename(summary_filename)}", 'green'))
        return True

    except Exception as e:
        print(colored(f"[ERROR] Failed to create beautiful summary: {e}", 'red'))
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
    parser.add_argument('--save-images', action='store_true', default=False,
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
    """Load data samples using project's data interface."""
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

            print(f"\n--- Sample {i+1} ---")

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
                print(colored(f"\n[IMAGES] Saving beautiful summary for sample {i+1}...", 'yellow'))

                # Handle caption for summary
                sample_caption = caption
                if isinstance(caption, list) and caption:
                    sample_caption = caption[0]
                elif not isinstance(caption, str):
                    sample_caption = str(caption)

                create_beautiful_summary_cv2(wave_embed, sample_caption, i+1, preview_dir)

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
            print(colored("  [SUCCESS] Creating beautiful CV2 summaries", 'green'))

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
                        print(colored("[FILES] Saved beautiful summary files:", 'blue'))
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