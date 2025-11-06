#!/usr/bin/env python3
"""
CLIP Model t-SNE Visualization Script

This script loads a trained CLIP model and performs t-SNE visualization
on features extracted from test data items containing specified words.
"""

import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
from pathlib import Path
import yaml
import glob

# Add project root to path
sys.path.append('/root/autodl-tmp/mmExpert')

from src.clip import CLIP
from src.clip.datamodule import HumanDInterface
from src.clip.utils.io import load_config
from src.clip.core.base import ModalityType


class CLIPTSNEVisualizer:
    def __init__(self, model_path, config_path=None, device='cuda', batch_size=64):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.batch_size = batch_size
        
        # Auto-discover config if not provided
        if self.config_path is None:
            self.config_path = self.discover_config_path()
        
        # Load model
        self.model = self.load_model()
        self.model.eval()
        self.model.to(device)
        
        # Load test data
        self.test_dataloader = self.load_test_data()
        
    def discover_config_path(self):
        """Auto-discover config file based on checkpoint location"""
        model_dir = os.path.dirname(self.model_path)
        
        # Extract version name from path
        if os.path.basename(model_dir) == 'checkpoints':
            version_dir = os.path.dirname(model_dir)
            version_name = os.path.basename(version_dir)
            # Try config directory in version directory first
            config_dir = os.path.join(version_dir, 'config')
            if not os.path.exists(config_dir):
                config_dir = os.path.join('config', version_name)
        else:
            version_name = os.path.basename(model_dir)
            config_dir = os.path.join('config', version_name)
            if not os.path.exists(config_dir):
                config_dir = model_dir
        
        # Look for config files
        model_config = os.path.join(config_dir, 'model_config.yaml')
        data_config = os.path.join(config_dir, 'data_config.yaml')
        single_config = os.path.join(config_dir, 'config.yaml')
        
        if os.path.exists(model_config) and os.path.exists(data_config):
            return self.merge_configs(model_config, data_config)
        elif os.path.exists(single_config):
            return single_config
        elif os.path.exists(model_config):
            return model_config
        elif os.path.exists(data_config):
            return data_config
        else:
            # Try checkpoint directory
            model_config_old = os.path.join(model_dir, 'model_config.yaml')
            data_config_old = os.path.join(model_dir, 'data_config.yaml')
            if os.path.exists(model_config_old) and os.path.exists(data_config_old):
                return self.merge_configs(model_config_old, data_config_old)
            elif os.path.exists(model_config_old):
                return model_config_old
            else:
                raise FileNotFoundError(f"Config file not found for model: {self.model_path}")
    
    def merge_configs(self, model_config_path, data_config_path):
        """Merge model and data configs"""
        import tempfile
        
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        merged_config = {
            'model_cfg': model_config,
            'data_cfg': data_config
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_merged_config.yaml', delete=False)
        yaml.dump(merged_config, temp_file, default_flow_style=False)
        temp_file.close()
        
        return temp_file.name
    
    def load_model(self):
        """Load trained CLIP model"""
        print(f"Loading model from {self.model_path}")
        
        config = load_config(self.config_path)
        model_cfg = config['model_cfg']['params']
        model = CLIP(**model_cfg)
        
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
        return model
    
    def load_test_data(self):
        """Load test dataset"""
        print("Loading test dataset...")
        
        config = load_config(self.config_path)
        data_cfg = config['data_cfg']
        
        # Override batch size
        original_batch_size = data_cfg['params']['cfg']['batch_size']
        data_cfg['params']['cfg']['batch_size'] = self.batch_size
        
        data_interface = HumanDInterface(data_cfg['params']['cfg'])
        data_interface.setup('test')
        
        test_dataloader = data_interface.test_dataloader()
        print(f"Test dataset loaded: {len(test_dataloader.dataset)} samples")
        
        return test_dataloader
    
    def filter_data_by_words(self, words):
        """Filter test data items containing any of the specified words"""
        words_str = "', '".join(words)
        print(f"Filtering data items containing any of: '{words_str}'...")
        
        filtered_indices = []
        filtered_captions = []
        
        dataset = self.test_dataloader.dataset
        words_lower = [w.lower() for w in words]
        
        for idx in tqdm(range(len(dataset)), desc="Filtering data"):
            try:
                item = dataset[idx]
                caption = item.get('caption', '')
                if not isinstance(caption, str):
                    caption = str(caption)
                caption_lower = caption.lower()
                
                # Check if caption contains any of the words
                if any(word in caption_lower for word in words_lower):
                    filtered_indices.append(idx)
                    filtered_captions.append(caption)
            except Exception as e:
                print(f"Warning: Error loading item {idx}: {e}")
                continue
        
        print(f"Found {len(filtered_indices)} data items containing any of: '{words_str}'")
        return filtered_indices, filtered_captions
    
    def extract_features_for_indices(self, indices, words):
        """Extract features for specific data indices"""
        print(f"Extracting features for {len(indices)} data items...")
        
        radar_features_list = []
        text_features_list = []
        labels_list = []
        captions_list = []
        
        dataset = self.test_dataloader.dataset
        words_lower = [w.lower() for w in words]
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Extracting features"):
                try:
                    item = dataset[idx]
                    caption = item.get('caption', '')
                    if not isinstance(caption, str):
                        caption = str(caption)
                    
                    # Prepare radar data - only include non-None views
                    radar_data = {}
                    if item.get('input_wave_range') is not None:
                        radar_data['range_time'] = torch.tensor(item['input_wave_range']).unsqueeze(0).float()
                    if item.get('input_wave_doppler') is not None:
                        radar_data['doppler_time'] = torch.tensor(item['input_wave_doppler']).unsqueeze(0).float()
                    if item.get('input_wave_azimuth') is not None:
                        radar_data['azimuth_time'] = torch.tensor(item['input_wave_azimuth']).unsqueeze(0).float()
                    
                    # Check if we have at least one view
                    if len(radar_data) == 0:
                        raise ValueError(f"No radar data available for item {idx}")
                    
                    # Move to device
                    for key in radar_data:
                        radar_data[key] = radar_data[key].to(self.device)
                    
                    # Encode data
                    encoding_results = self.model._encode_data(
                        radar_data=radar_data,
                        text_data=[caption]
                    )
                    
                    # Extract features
                    radar_feat = encoding_results[ModalityType.RADAR].features
                    text_feat = encoding_results[ModalityType.TEXT].features
                    
                    # Normalize features
                    radar_feat = radar_feat / radar_feat.norm(dim=1, keepdim=True)
                    text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
                    
                    # Convert to numpy
                    radar_features_list.append(radar_feat.cpu().numpy()[0])
                    text_features_list.append(text_feat.cpu().numpy()[0])
                    captions_list.append(caption)
                    
                    # Determine label based on words in caption
                    caption_lower = caption.lower()
                    matched_words = [i for i, word in enumerate(words_lower) if word in caption_lower]
                    
                    if len(matched_words) == 0:
                        labels_list.append('other')
                    elif len(matched_words) == 1:
                        # Only one word matched
                        labels_list.append(f'word{matched_words[0]}')
                    else:
                        # Multiple words matched - use the first one
                        labels_list.append(f'word{matched_words[0]}')
                        
                except Exception as e:
                    print(f"Warning: Error processing item {idx}: {e}")
                    continue
        
        radar_features = np.array(radar_features_list)
        text_features = np.array(text_features_list)
        
        print(f"Extracted features: radar shape {radar_features.shape}, text shape {text_features.shape}")
        
        return radar_features, text_features, labels_list, captions_list
    
    def visualize_tsne(self, features, labels, words, modality_type, output_path):
        """Perform t-SNE visualization for N words"""
        print(f"Performing t-SNE for {modality_type} features...")
        
        # Create word label mapping
        word_labels = [f'word{i}' for i in range(len(words))]
        
        # Filter to only show word labels (exclude 'other')
        mask_only = np.array([label in word_labels for label in labels])
        features_filtered = features[mask_only]
        labels_filtered = [labels[i] for i in range(len(labels)) if mask_only[i]]
        
        if len(features_filtered) == 0:
            words_str = "', '".join(words)
            print(f"Warning: No data points found for words: '{words_str}'")
            return
        
        words_str = "', '".join(words)
        print(f"Visualizing {len(features_filtered)} data points for words: '{words_str}'")
        
        # Adjust perplexity based on sample size (must be less than n_samples)
        n_samples = len(features_filtered)
        perplexity = min(30, max(5, n_samples - 1))
        print(f"Using perplexity={perplexity} for {n_samples} samples")
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        features_2d = tsne.fit_transform(features_filtered)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Generate colors for N words using colormap
        n_words = len(words)
        if n_words <= 10:
            # Use predefined colors for small number of words
            predefined_colors = ['red', 'blue', 'green', 'orange', 'purple', 
                               'brown', 'pink', 'gray', 'olive', 'cyan']
            colors = {f'word{i}': predefined_colors[i % len(predefined_colors)] 
                     for i in range(n_words)}
        else:
            # Use colormap for many words
            try:
                colormap = plt.cm.get_cmap('tab20')
            except AttributeError:
                # For newer matplotlib versions
                colormap = plt.colormaps['tab20']
            colors = {f'word{i}': colormap(i / max(n_words - 1, 1)) 
                     for i in range(n_words)}
        
        # Plot each word category
        for i, word in enumerate(words):
            label = f'word{i}'
            label_name = f'Only "{word}"'
            mask = np.array(labels_filtered) == label
            if np.any(mask):
                color = colors[label]
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=color, label=label_name,
                           alpha=0.6, s=50)
        
        # Create title with all words
        words_title = " vs ".join([f'"{w}"' for w in words])
        plt.title(f't-SNE Visualization of {modality_type.upper()} Features\n'
                 f'Words: {words_title}', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE visualization to {output_path}")
        plt.close()
    
    def run_visualization(self, words, output_dir='./tmp'):
        """Run complete t-SNE visualization pipeline"""
        # Validate input
        if not isinstance(words, list) or len(words) == 0:
            raise ValueError("words must be a non-empty list")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter data by words
        indices, captions = self.filter_data_by_words(words)
        
        if len(indices) == 0:
            words_str = "', '".join(words)
            print(f"Error: No data items found containing any of: '{words_str}'")
            return
        
        # Extract features
        radar_features, text_features, labels, captions_list = self.extract_features_for_indices(indices, words)
        
        # Sanitize words for filename
        def sanitize_filename(text):
            return "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        
        words_safe = '_'.join([sanitize_filename(w) for w in words])
        
        # Visualize radar features
        radar_output_path = os.path.join(output_dir, f'tsne_radar_{words_safe}.png')
        self.visualize_tsne(radar_features, labels, words, 'radar', radar_output_path)
        
        # Visualize text features
        text_output_path = os.path.join(output_dir, f'tsne_text_{words_safe}.png')
        self.visualize_tsne(text_features, labels, words, 'text', text_output_path)
        
        print(f"\n✅ Visualization complete!")
        print(f"  - Radar t-SNE: {radar_output_path}")
        print(f"  - Text t-SNE: {text_output_path}")
        print(f"  - Total items: {len(indices)}")
        print(f"  - Words: {', '.join(words)}")


def main():
    parser = argparse.ArgumentParser(description='CLIP t-SNE Visualization')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config file (auto-discovered if not provided)')
    parser.add_argument('--words', type=str, nargs='+', required=True,
                       help='Words to filter data items (can specify multiple words, e.g., --words walk run sit)')
    parser.add_argument('--output_dir', type=str, default='./tmp',
                       help='Output directory for visualization results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for data loading')
    
    args = parser.parse_args()
    
    # Convert words to list if it's a single string
    words = args.words if isinstance(args.words, list) else [args.words]
    
    # Create visualizer
    visualizer = CLIPTSNEVisualizer(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run visualization
    visualizer.run_visualization(words, args.output_dir)


if __name__ == "__main__":
    main()

