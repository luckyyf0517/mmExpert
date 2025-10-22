#!/usr/bin/env python3
"""
CLIP Model Evaluation Script

This script evaluates trained CLIP models on test set using:
1. Radar→Text retrieval
2. Text→Radar retrieval
3. Recall@K metrics
4. Top-K accuracy
"""

import os
import sys
import torch
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
import argparse
from pathlib import Path
import yaml
import glob

# Add project root to path
sys.path.append('/root/autodl-tmp/mmExpert')

from src.model.clip import CLIP
from src.data_interface import HumanDInterface
from src.misc.io import load_config
from easydict import EasyDict as edict
import pytorch_lightning as pl

class CLIPEvaluator:
    def __init__(self, model_path=None, config_path=None, version_path=None, device='cuda', debug=False, batch_size=None):
        self.device = device
        self.model_path = model_path
        self.config_path = config_path
        self.version_path = version_path
        self.debug = debug
        self.override_batch_size = batch_size

        # Handle version path option
        if self.version_path is not None:
            self.model_path, self.config_path = self.discover_from_version(self.version_path)

        # Auto-discover config if not provided
        if self.config_path is None:
            self.config_path = self.discover_config_path()

        # Load model
        self.model = self.load_model()
        self.model.eval()
        self.model.to(device)

        # Load test data
        self.test_dataloader = self.load_test_data()

    def discover_from_version(self, version_path):
        """Discover model and config paths from version directory"""
        print(f"🔍 Discovering files from version path: {version_path}")

        # Resolve version path - could be experiment name or full path
        if not os.path.isabs(version_path) and '/' not in version_path:
            # Assume it's just the experiment name
            version_path = os.path.join('log/humanml3d_experiments-text-encoder', version_path)

        if not os.path.exists(version_path):
            raise FileNotFoundError(f"Version directory not found: {version_path}")

        # Find checkpoint directory
        checkpoints_dir = os.path.join(version_path, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

        # Look for last.ckpt first, then fall back to other checkpoints
        last_ckpt = os.path.join(checkpoints_dir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            model_path = last_ckpt
            print(f"✅ Found last checkpoint: {model_path}")
        else:
            # Find any checkpoint file
            ckpt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoint files found in: {checkpoints_dir}")

            # Sort by epoch number if possible, otherwise use the first one
            ckpt_files.sort()
            model_path = os.path.join(checkpoints_dir, ckpt_files[-1])
            print(f"✅ Found checkpoint: {model_path}")

        # Find config directory
        version_name = os.path.basename(version_path)
        config_dir = os.path.join('config', version_name)
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        print(f"✅ Using config directory: {config_dir}")

        # Let the existing discover_config_path handle the config merging
        return model_path, None  # config_path will be auto-discovered

    def discover_config_path(self):
        """Auto-discover and merge config files based on checkpoint location"""
        model_dir = os.path.dirname(self.model_path)

        # Extract version name from path to find config directory
        # Path could be: log/version/checkpoints/model.ckpt or log/version/model.ckpt (old format)
        if os.path.basename(model_dir) == 'checkpoints':
            # New format: log/version/checkpoints/model.ckpt
            version_dir = os.path.dirname(model_dir)
            version_name = os.path.basename(version_dir)
            # First try: log/version/config/ (newest format)
            config_dir = os.path.join(version_dir, 'config')
            # Fallback: config/version/ (old format)
            if not os.path.exists(config_dir):
                config_dir = os.path.join('config', version_name)
        else:
            # Old format: log/version/model.ckpt
            version_name = os.path.basename(model_dir)
            config_dir = os.path.join('config', version_name)
            # Also check the same directory as the checkpoint for backwards compatibility
            if not os.path.exists(config_dir):
                config_dir = model_dir

        # Look for config files in the config directory first (new format)
        model_config = os.path.join(config_dir, 'model_config.yaml')
        data_config = os.path.join(config_dir, 'data_config.yaml')
        single_config = os.path.join(config_dir, 'config.yaml')

        # Check for different config patterns in config directory
        has_model_config = os.path.exists(model_config)
        has_data_config = os.path.exists(data_config)
        has_single_config = os.path.exists(single_config)

        # If not found in config directory, check the checkpoint directory (old format)
        if not has_model_config and not has_data_config and not has_single_config:
            model_config_old = os.path.join(model_dir, 'model_config.yaml')
            data_config_old = os.path.join(model_dir, 'data_config.yaml')
            single_config_old = os.path.join(model_dir, 'config.yaml')

            has_model_config_old = os.path.exists(model_config_old)
            has_data_config_old = os.path.exists(data_config_old)
            has_single_config_old = os.path.exists(single_config_old)

            if has_model_config_old or has_data_config_old or has_single_config_old:
                # Use old format
                model_config = model_config_old
                data_config = data_config_old
                single_config = single_config_old
                has_model_config = has_model_config_old
                has_data_config = has_data_config_old
                has_single_config = has_single_config_old

        if not has_model_config and not has_data_config and not has_single_config:
            # Look for any yaml files in both directories
            yaml_files = glob.glob(os.path.join(config_dir, '*.yaml'))
            yaml_files.extend(glob.glob(os.path.join(model_dir, '*.yaml')))
            if not yaml_files:
                raise FileNotFoundError(
                    f"No config files found in {config_dir} or {model_dir}. "
                    f"Please specify --config_path manually."
                )
            return yaml_files[0]  # Return the first found

        # If we have separate model and data configs, merge them
        if has_model_config and has_data_config:
            return self.merge_configs(model_config, data_config)
        elif has_single_config:
            print(f"Auto-discovered config: {single_config}")
            return single_config
        elif has_model_config:
            print(f"Auto-discovered model config: {model_config}")
            return model_config
        else:  # has_data_config only
            print(f"Auto-discovered data config: {data_config}")
            return data_config

    def merge_configs(self, model_config_path, data_config_path):
        """Merge model and data configs into a single config for evaluation"""
        import tempfile

        # Load both configs
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)

        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Create merged config
        merged_config = {
            'model_cfg': model_config,
            'data_cfg': data_config
        }

        # Create temporary file for merged config
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_merged_config.yaml', delete=False)
        yaml.dump(merged_config, temp_file, default_flow_style=False)
        temp_file.close()

        print(f"Auto-discovered and merged configs:")
        print(f"  Model config: {model_config_path}")
        print(f"  Data config: {data_config_path}")
        print(f"  Merged config: {temp_file.name}")

        return temp_file.name

    def load_model(self):
        """Load trained CLIP model"""
        print(f"Loading model from {self.model_path}")

        # Load config using YAML loader
        config = load_config(self.config_path)

        model_cfg = config['model_cfg']['params']
        model = CLIP(**model_cfg)

        # Load checkpoint with weights_only=False to handle EasyDict
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"Model loaded successfully")
        return model

    def load_test_data(self):
        """Load test dataset"""
        print("Loading test dataset...")

        # Load config using YAML loader
        config = load_config(self.config_path)

        data_cfg = config['data_cfg']

        # Override batch size if specified
        if self.override_batch_size is not None:
            original_batch_size = data_cfg['params']['cfg']['batch_size']
            data_cfg['params']['cfg']['batch_size'] = self.override_batch_size
            print(f"Overriding batch size: {original_batch_size} → {self.override_batch_size}")

        data_interface = HumanDInterface(data_cfg['params']['cfg'])
        data_interface.setup('test')

        test_dataloader = data_interface.test_dataloader()
        print(f"Test dataset loaded: {len(test_dataloader.dataset)} samples")
        print(f"Using batch size: {test_dataloader.batch_size}")

        return test_dataloader

    def extract_features(self):
        """Extract features from test dataset (batch-wise evaluation)"""
        print("Extracting features and evaluating batch-wise...")

        batch_size = self.test_dataloader.batch_size
        all_batch_results = []
        all_captions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_dataloader, desc="Batch evaluation")):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Prepare radar data - use correct key names that radar encoder expects
                radar_data = {
                    'range_time': batch['input_wave_range'],
                    'doppler_time': batch['input_wave_doppler'],
                    'azimuth_time': batch['input_wave_azimuth']
                }

                # Get features using new unified encoding method
                encoding_results = self.model._encode_data(
                    radar_data=radar_data,
                    text_data=batch['caption']
                )

                # Extract features from encoding results
                from src.core.base import ModalityType
                radar_features = encoding_results[ModalityType.RADAR].features
                text_features = encoding_results[ModalityType.TEXT].features

                # Normalize features
                radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # Get captions for this batch
                batch_captions = batch['caption']

                # Evaluate batch-wise retrieval with debug info
                batch_results = self.evaluate_batch_retrieval(
                    radar_features, text_features, batch_idx, batch_captions
                )
                all_batch_results.append(batch_results)
                all_captions.extend(batch_captions)

        # Aggregate results across all batches
        aggregated_results = self.aggregate_batch_results(all_batch_results)

        print(f"Batch-wise evaluation completed:")
        print(f"  Total batches: {len(all_batch_results)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total samples evaluated: {len(all_captions)}")

        return aggregated_results, all_captions

    def evaluate_batch_retrieval(self, radar_features, text_features, batch_idx, captions=None):
        """Evaluate retrieval within a batch (in-batch negative sampling)"""
        batch_size = len(radar_features)

        # Compute similarity matrix within batch
        similarity_matrix = torch.matmul(radar_features, text_features.T)

        batch_results = {
            'batch_size': batch_size,
            'batch_idx': batch_idx,
            'r2t_correct': [],
            't2r_correct': [],
            'r2t_ranks': [],
            't2r_ranks': [],
            'similarities_diag': [],
            'similarities_off_diag': []
        }

        # Debug output for this batch
        if self.debug and captions is not None:
            print(f"\n" + "="*80)
            print(f"🔍 DEBUG - Batch {batch_idx} (Size: {batch_size})")
            print(f"="*80)

        # Evaluate Radar→Text retrieval
        for i in range(batch_size):
            # Get sorted indices for this radar sample
            _, sorted_indices = torch.sort(similarity_matrix[i], descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            batch_results['r2t_ranks'].append(rank)
            batch_results['r2t_correct'].append(rank == 1)  # Recall@1
            batch_results['similarities_diag'].append(similarity_matrix[i, i].item())

            # Debug output for this sample
            if self.debug and captions is not None:
                correct_caption = captions[i]
                predicted_idx = sorted_indices[0].item()
                predicted_caption = captions[predicted_idx]
                similarity_score = similarity_matrix[i, predicted_idx].item()

                print(f"\n📊 Sample {i:2d}:")
                print(f"   ✅ Correct Caption: {correct_caption}")
                print(f"   🎯 Predicted Caption: {predicted_caption}")
                print(f"   📈 Similarity Score: {similarity_score:.4f}")
                print(f"   🏆 Rank of Correct: {rank}")

                # Show top 3 predictions
                print(f"   🔝 Top 3 Predictions:")
                for j in range(min(3, batch_size)):
                    pred_idx = sorted_indices[j].item()
                    pred_caption = captions[pred_idx]
                    pred_score = similarity_matrix[i, pred_idx].item()
                    is_correct = "✅" if pred_idx == i else "❌"
                    print(f"      {j+1}. {is_correct} [{pred_score:.4f}] {pred_caption}")

        # Evaluate Text→Radar retrieval
        for i in range(batch_size):
            # Get sorted indices for this text sample
            _, sorted_indices = torch.sort(similarity_matrix[:, i], descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            batch_results['t2r_ranks'].append(rank)
            batch_results['t2r_correct'].append(rank == 1)  # Recall@1
            # Sample some off-diagonal similarities for analysis
            for j in range(batch_size):
                if i != j:
                    batch_results['similarities_off_diag'].append(similarity_matrix[j, i].item())
                    break  # Just take one off-diagonal per sample

        return batch_results

    def aggregate_batch_results(self, all_batch_results):
        """Aggregate results across all batches"""
        total_samples = sum(result['batch_size'] for result in all_batch_results)

        # Collect all metrics
        all_r2t_correct = []
        all_t2r_correct = []
        all_r2t_ranks = []
        all_t2r_ranks = []
        all_diag_similarities = []
        all_off_diag_similarities = []

        for batch_result in all_batch_results:
            all_r2t_correct.extend(batch_result['r2t_correct'])
            all_t2r_correct.extend(batch_result['t2r_correct'])
            all_r2t_ranks.extend(batch_result['r2t_ranks'])
            all_t2r_ranks.extend(batch_result['t2r_ranks'])
            all_diag_similarities.extend(batch_result['similarities_diag'])
            all_off_diag_similarities.extend(batch_result['similarities_off_diag'])

        # Calculate aggregated metrics
        results = {}

        # Recall@K metrics
        results['radar_to_text_recall@1'] = np.mean(all_r2t_correct)
        results['text_to_radar_recall@1'] = np.mean(all_t2r_correct)

        # Recall@5 and @10
        results['radar_to_text_recall@5'] = np.mean([rank <= 5 for rank in all_r2t_ranks])
        results['text_to_radar_recall@5'] = np.mean([rank <= 5 for rank in all_t2r_ranks])
        results['radar_to_text_recall@10'] = np.mean([rank <= 10 for rank in all_r2t_ranks])
        results['text_to_radar_recall@10'] = np.mean([rank <= 10 for rank in all_t2r_ranks])

        # MRR
        results['radar_to_text_mrr'] = np.mean([1.0/rank for rank in all_r2t_ranks])
        results['text_to_radar_mrr'] = np.mean([1.0/rank for rank in all_t2r_ranks])

        # Median Rank
        results['radar_to_text_median_rank'] = np.median(all_r2t_ranks)
        results['text_to_radar_median_rank'] = np.median(all_t2r_ranks)

        # Classification accuracy (same as Recall@1)
        results['radar_to_text_accuracy'] = results['radar_to_text_recall@1']
        results['text_to_radar_accuracy'] = results['text_to_radar_recall@1']

        # Similarity statistics
        results['diag_similarity_mean'] = np.mean(all_diag_similarities)
        results['diag_similarity_std'] = np.std(all_diag_similarities)
        results['off_diag_similarity_mean'] = np.mean(all_off_diag_similarities)
        results['off_diag_similarity_std'] = np.std(all_off_diag_similarities)
        results['separation'] = results['diag_similarity_mean'] - results['off_diag_similarity_mean']

        return results

    def analyze_similarity_distribution(self, similarity_matrix):
        """Analyze similarity score distribution"""
        print("\n5. Similarity Score Distribution:")

        # Get diagonal (correct pairs) and off-diagonal (incorrect pairs)
        diagonal_scores = torch.diag(similarity_matrix).cpu().numpy()

        # Get random off-diagonal samples for comparison
        n_samples = len(diagonal_scores)
        off_diagonal_scores = []

        for i in range(min(1000, n_samples * 10)):  # Sample up to 1000 pairs
            row = np.random.randint(0, n_samples)
            col = np.random.randint(0, n_samples)
            if row != col:
                off_diagonal_scores.append(similarity_matrix[row, col].item())

        off_diagonal_scores = np.array(off_diagonal_scores)

        print(f"  Correct pairs (diagonal):")
        print(f"    Mean: {np.mean(diagonal_scores):.4f}")
        print(f"    Std:  {np.std(diagonal_scores):.4f}")
        print(f"    Min:  {np.min(diagonal_scores):.4f}")
        print(f"    Max:  {np.max(diagonal_scores):.4f}")

        print(f"  Incorrect pairs (off-diagonal):")
        print(f"    Mean: {np.mean(off_diagonal_scores):.4f}")
        print(f"    Std:  {np.std(off_diagonal_scores):.4f}")
        print(f"    Min:  {np.min(off_diagonal_scores):.4f}")
        print(f"    Max:  {np.max(off_diagonal_scores):.4f}")

        # Separation metric
        separation = np.mean(diagonal_scores) - np.mean(off_diagonal_scores)
        print(f"  Separation (correct - incorrect): {separation:.4f}")

    def evaluate_classification_accuracy(self, similarity_matrix):
        """Evaluate classification accuracy (1-to-1 matching)"""
        print("\n6. Classification Accuracy:")

        # For each radar sample, find the most similar text
        _, predicted_text_indices = torch.max(similarity_matrix, dim=1)
        correct_indices = torch.arange(len(similarity_matrix), device=self.device)

        accuracy = (predicted_text_indices == correct_indices).float().mean().item()
        print(f"  Radar→Text Classification Accuracy: {accuracy:.4f}")

        # For each text sample, find the most similar radar
        _, predicted_radar_indices = torch.max(similarity_matrix.T, dim=1)
        accuracy_reverse = (predicted_radar_indices == correct_indices).float().mean().item()
        print(f"  Text→Radar Classification Accuracy: {accuracy_reverse:.4f}")

        return accuracy, accuracy_reverse

    def save_results(self, results, output_path):
        """Save evaluation results"""
        results_data = {
            'model_path': self.model_path,
            'config_path': self.config_path,
            'results': results,
            'dataset_size': len(self.test_dataloader.dataset)
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def run_evaluation(self, output_path=None):
        """Run complete evaluation with batch-wise negative sampling"""
        print("=" * 60)
        print("CLIP Model Evaluation (Batch-wise)")
        print("=" * 60)
        print(f"Batch Size: {self.test_dataloader.batch_size}")
        print(f"Evaluation Mode: {'DEBUG - Detailed Output' if self.debug else 'Standard - Summary Only'}")

        # Extract features and evaluate batch-wise
        results, captions = self.extract_features()

        # Print batch-wise evaluation results
        print(f"\n📊 Batch-wise Evaluation Results (Batch Size: {self.test_dataloader.batch_size}):")
        print(f"  Radar → Text:")
        print(f"    Recall@1:  {results['radar_to_text_recall@1']:.4f}")
        print(f"    Recall@5:  {results['radar_to_text_recall@5']:.4f}")
        print(f"    Recall@10: {results['radar_to_text_recall@10']:.4f}")
        print(f"    MRR:        {results['radar_to_text_mrr']:.4f}")
        print(f"    Median Rank: {results['radar_to_text_median_rank']:.1f}")

        print(f"  Text → Radar:")
        print(f"    Recall@1:  {results['text_to_radar_recall@1']:.4f}")
        print(f"    Recall@5:  {results['text_to_radar_recall@5']:.4f}")
        print(f"    Recall@10: {results['text_to_radar_recall@10']:.4f}")
        print(f"    MRR:        {results['text_to_radar_mrr']:.4f}")
        print(f"    Median Rank: {results['text_to_radar_median_rank']:.1f}")

        print(f"\n📈 Similarity Analysis:")
        print(f"  Correct pairs (diagonal):")
        print(f"    Mean: {results['diag_similarity_mean']:.4f} ± {results['diag_similarity_std']:.4f}")
        print(f"  Incorrect pairs (off-diagonal):")
        print(f"    Mean: {results['off_diag_similarity_mean']:.4f} ± {results['off_diag_similarity_std']:.4f}")
        print(f"  Separation: {results['separation']:.4f}")

        # Add classification accuracy
        results['radar_to_text_accuracy'] = results['radar_to_text_recall@1']
        results['text_to_radar_accuracy'] = results['text_to_radar_recall@1']

        # Save results
        if output_path:
            self.save_results(results, output_path)

        print("\n" + "=" * 60)
        print("Batch-wise Evaluation Complete!")
        print("=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIP model')

    # Create mutually exclusive group for model specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    group.add_argument('--version', type=str, default=None,
                       help='Version name or path to experiment directory. Will auto-discover last checkpoint and config files')

    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to model configuration file (YAML format). If not specified, will auto-discover in checkpoint directory')
    parser.add_argument('--output_path', type=str, default='clip_evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to show detailed prediction results for each batch')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size for evaluation (default: use config file value)')

    args = parser.parse_args()

    # Create evaluator
    evaluator = CLIPEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        version_path=args.version,
        device=args.device,
        debug=args.debug,
        batch_size=args.batch_size
    )

    # Run evaluation
    results = evaluator.run_evaluation(output_path=args.output_path)

    if not args.debug:  # Only show summary if not in debug mode (debug shows detailed info)
        print(f"\n📊 Final Results Summary:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()