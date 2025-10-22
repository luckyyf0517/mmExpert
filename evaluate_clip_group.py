#!/usr/bin/env python3
"""
CLIP Group Evaluation Script

This script evaluates multiple CLIP models in a group directory using:
1. Auto-discovery of all versions
2. Individual model evaluation
3. Aggregated results with formatted comparison
4. Summary statistics across all models
"""

import os
import sys
import torch
import numpy as np
import json
import yaml
import argparse
from pathlib import Path
import glob
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate

# Add project root to path
sys.path.append('/root/autodl-tmp/mmExpert')

from evaluate_clip import CLIPEvaluator


class CLIPGroupEvaluator:
    def __init__(self, group_dir, device='cuda', batch_size=None, verbose=False):
        self.group_dir = group_dir
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        # Discover all versions
        self.versions = self.discover_versions()
        print(f"🔍 Discovered {len(self.versions)} versions in {group_dir}")

        # Results storage
        self.results = {}

    def discover_versions(self):
        """Discover all version directories in the group folder"""
        versions = []

        if not os.path.exists(self.group_dir):
            raise FileNotFoundError(f"Group directory not found: {self.group_dir}")

        # Look for subdirectories that contain checkpoints
        for item in os.listdir(self.group_dir):
            version_path = os.path.join(self.group_dir, item)
            if os.path.isdir(version_path):
                checkpoints_dir = os.path.join(version_path, 'checkpoints')
                if os.path.exists(checkpoints_dir):
                    # Check if there are any checkpoint files
                    ckpt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
                    if ckpt_files:
                        versions.append({
                            'name': item,
                            'path': version_path,
                            'checkpoints_dir': checkpoints_dir,
                            'ckpt_files': ckpt_files
                        })

        return sorted(versions, key=lambda x: x['name'])

    def load_model_and_config(self, version):
        """Load model and config for a specific version"""
        version_name = version['name']
        version_path = version['path']

        # Find last.ckpt or fallback to other checkpoint
        checkpoints_dir = version['checkpoints_dir']
        last_ckpt = os.path.join(checkpoints_dir, 'last.ckpt')

        if os.path.exists(last_ckpt):
            model_path = last_ckpt
        else:
            # Use the last checkpoint alphabetically
            ckpt_files = sorted(version['ckpt_files'])
            model_path = os.path.join(checkpoints_dir, ckpt_files[-1])

        # Find config files
        config_dir = os.path.join('config', version_name)
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        model_config = os.path.join(config_dir, 'model_config.yaml')
        data_config = os.path.join(config_dir, 'data_config.yaml')

        if not os.path.exists(model_config) or not os.path.exists(data_config):
            raise FileNotFoundError(f"Config files not found for {version_name}")

        # Load and merge configs
        with open(model_config, 'r') as f:
            model_cfg_data = yaml.safe_load(f)
        with open(data_config, 'r') as f:
            data_cfg_data = yaml.safe_load(f)

        merged_config = {
            'model_cfg': model_cfg_data,
            'data_cfg': data_cfg_data
        }

        return model_path, merged_config

    def evaluate_single_model(self, version):
        """Evaluate a single CLIP model using CLIPEvaluator"""
        version_name = version['name']

        try:
            if self.verbose:
                print(f"\n📊 Evaluating version: {version_name}")
            else:
                print(f"🔄 Evaluating: {version_name}", end=' ... ')

            # Find last.ckpt or fallback to other checkpoint
            checkpoints_dir = version['checkpoints_dir']
            last_ckpt = os.path.join(checkpoints_dir, 'last.ckpt')

            if os.path.exists(last_ckpt):
                model_path = last_ckpt
            else:
                # Use the last checkpoint alphabetically
                ckpt_files = sorted(version['ckpt_files'])
                model_path = os.path.join(checkpoints_dir, ckpt_files[-1])

            # Use CLIPEvaluator for evaluation
            evaluator = CLIPEvaluator(
                model_path=model_path,
                device=self.device,
                debug=self.verbose,
                batch_size=self.batch_size
            )

            # Run evaluation and get results
            results = evaluator.run_evaluation()

            # Store results
            self.results[version_name] = {
                'model_path': model_path,
                'results': results,
                'status': 'success'
            }

            if not self.verbose:
                print("✅ Done")

        except Exception as e:
            error_msg = f"❌ Failed: {str(e)}"
            if self.verbose:
                print(error_msg)
            else:
                print(error_msg)

            self.results[version_name] = {
                'model_path': model_path if 'model_path' in locals() else None,
                'results': None,
                'status': 'failed',
                'error': str(e)
            }

    def evaluate_all(self):
        """Evaluate all models in the group"""
        print(f"\n🚀 Starting group evaluation for {len(self.versions)} models...")
        print("=" * 80)

        for version in self.versions:
            self.evaluate_single_model(version)

        print("\n" + "=" * 80)
        print("✅ Group evaluation complete!")

    def format_results(self):
        """Format results for comparison display"""
        successful_results = {k: v for k, v in self.results.items() if v['status'] == 'success'}

        if not successful_results:
            print("❌ No successful evaluations to display!")
            return

        # Create comparison table
        metrics = [
            'radar_to_text_recall@1', 'text_to_radar_recall@1',
            'radar_to_text_recall@5', 'text_to_radar_recall@5',
            'radar_to_text_recall@10', 'text_to_radar_recall@10',
            'radar_to_text_mrr', 'text_to_radar_mrr',
            'radar_to_text_median_rank', 'text_to_radar_median_rank',
            'separation'
        ]

        # Prepare data for table
        table_data = []
        headers = ['Version'] + [m.replace('_', '\n') for m in metrics]

        for version_name, data in successful_results.items():
            row = [version_name]
            results = data['results']
            for metric in metrics:
                value = results.get(metric, 0)
                if 'recall' in metric or 'mrr' in metric:
                    row.append(f"{value:.3f}")
                elif 'median_rank' in metric:
                    row.append(f"{value:.1f}")
                else:  # separation
                    row.append(f"{value:.4f}")
            table_data.append(row)

        # Sort by radar_to_text_recall@1 (best first)
        table_data.sort(key=lambda x: float(x[1]), reverse=True)

        # Display table
        print("\n📊 CLIP MODEL COMPARISON RESULTS")
        print("=" * 100)
        print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt=".3f"))

        # Summary statistics
        print("\n📈 SUMMARY STATISTICS")
        print("-" * 50)

        all_r2t_r1 = [data['results']['radar_to_text_recall@1'] for data in successful_results.values()]
        all_t2r_r1 = [data['results']['text_to_radar_recall@1'] for data in successful_results.values()]
        all_mrr = [data['results']['radar_to_text_mrr'] for data in successful_results.values()]

        print(f"Total models evaluated: {len(successful_results)}")
        print(f"Models failed: {len(self.results) - len(successful_results)}")
        print()
        print(f"Radar→Text Recall@1:")
        print(f"  Best: {max(all_r2t_r1):.3f}")
        print(f"  Worst: {min(all_r2t_r1):.3f}")
        print(f"  Average: {np.mean(all_r2t_r1):.3f} ± {np.std(all_r2t_r1):.3f}")
        print()
        print(f"Text→Radar Recall@1:")
        print(f"  Best: {max(all_t2r_r1):.3f}")
        print(f"  Worst: {min(all_t2r_r1):.3f}")
        print(f"  Average: {np.mean(all_t2r_r1):.3f} ± {np.std(all_t2r_r1):.3f}")
        print()
        print(f"MRR (Radar→Text):")
        print(f"  Best: {max(all_mrr):.3f}")
        print(f"  Worst: {min(all_mrr):.3f}")
        print(f"  Average: {np.mean(all_mrr):.3f} ± {np.std(all_mrr):.3f}")

        # Find best performing model
        best_model = max(successful_results.items(),
                        key=lambda x: x[1]['results']['radar_to_text_recall@1'])
        print(f"\n🏆 Best performing model: {best_model[0]}")
        print(f"   Radar→Text Recall@1: {best_model[1]['results']['radar_to_text_recall@1']:.3f}")
        print(f"   Text→Radar Recall@1: {best_model[1]['results']['text_to_radar_recall@1']:.3f}")

        # Show failed models
        failed_models = {k: v for k, v in self.results.items() if v['status'] == 'failed'}
        if failed_models:
            print(f"\n❌ Failed models:")
            for version_name, data in failed_models.items():
                print(f"   {version_name}: {data['error']}")

    def save_results(self, output_path):
        """Save detailed results to JSON file"""
        results_data = {
            'group_dir': self.group_dir,
            'evaluation_time': datetime.now().isoformat(),
            'device': self.device,
            'batch_size': self.batch_size,
            'total_versions': len(self.versions),
            'successful_evaluations': len([v for v in self.results.values() if v['status'] == 'success']),
            'failed_evaluations': len([v for v in self.results.values() if v['status'] == 'failed']),
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n💾 Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple CLIP models in a group')
    parser.add_argument('--group_dir', type=str, required=True,
                        help='Path to group directory containing version folders (e.g., log/humanml3d_experiments-text-encoder)')
    parser.add_argument('--output_path', type=str, default='/tmp/clip_group_evaluation_results.json',
                        help='Path to save detailed evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size for evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with detailed progress')

    args = parser.parse_args()

    # Create group evaluator
    evaluator = CLIPGroupEvaluator(
        group_dir=args.group_dir,
        device=args.device,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    # Run evaluation
    evaluator.evaluate_all()

    # Format and display results
    evaluator.format_results()

    # Save detailed results
    evaluator.save_results(args.output_path)


if __name__ == "__main__":
    main()