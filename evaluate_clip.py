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

# Add project root to path
sys.path.append('/root/autodl-tmp/mmExpert')

from src.model.clip import CLIP
from src.data_interface import HumanDInterface
from easydict import EasyDict as edict
import pytorch_lightning as pl

class CLIPEvaluator:
    def __init__(self, model_path, config_path, device='cuda'):
        self.device = device
        self.model_path = model_path
        self.config_path = config_path

        # Load model
        self.model = self.load_model()
        self.model.eval()
        self.model.to(device)

        # Load test data
        self.test_dataloader = self.load_test_data()

    def load_model(self):
        """Load trained CLIP model"""
        print(f"Loading model from {self.model_path}")

        # Load config
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        model_cfg = config['model_cfg']['params']
        model = CLIP(**model_cfg)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')

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

        # Load config
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        data_cfg = config['data_cfg']
        data_interface = HumanDInterface(data_cfg['params'])
        data_interface.setup('test')

        test_dataloader = data_interface.test_dataloader()
        print(f"Test dataset loaded: {len(test_dataloader.dataset)} samples")

        return test_dataloader

    def extract_features(self):
        """Extract features from test dataset"""
        print("Extracting features...")

        all_radar_features = []
        all_text_features = []
        all_captions = []

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Extracting features"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Get features
                radar_features = self.model.encode_radar(
                    batch['input_wave_range'],
                    batch['input_wave_doppler'],
                    batch['input_wave_azimuth']
                )

                text_features = self.model.encode_text(
                    batch['caption'], device=self.device
                )

                # Normalize features
                radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                all_radar_features.append(radar_features.cpu())
                all_text_features.append(text_features.cpu())
                all_captions.extend(batch['caption'])

        # Concatenate all features
        all_radar_features = torch.cat(all_radar_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

        print(f"Features extracted:")
        print(f"  Radar features: {all_radar_features.shape}")
        print(f"  Text features: {all_text_features.shape}")

        return all_radar_features, all_text_features, all_captions

    def evaluate_retrieval(self, radar_features, text_features, k_values=[1, 5, 10]):
        """Evaluate retrieval performance"""
        print("\nEvaluating retrieval performance...")

        # Move to device for computation
        radar_features = radar_features.to(self.device)
        text_features = text_features.to(self.device)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(radar_features, text_features.T)

        results = {}
        n_samples = len(radar_features)

        # Radar→Text retrieval
        print("\n1. Radar → Text Retrieval:")
        for k in k_values:
            # Get top-k predictions for each radar sample
            _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)

            # Calculate recall@k (assuming 1-to-1 pairing)
            correct_predictions = 0
            for i in range(n_samples):
                if i in top_k_indices[i]:  # Check if correct text is in top-k
                    correct_predictions += 1

            recall_k = correct_predictions / n_samples
            results[f'radar_to_text_recall@{k}'] = recall_k
            print(f"  Recall@{k}: {recall_k:.4f}")

        # Text→Radar retrieval
        print("\n2. Text → Radar Retrieval:")
        for k in k_values:
            # Get top-k predictions for each text sample
            _, top_k_indices = torch.topk(similarity_matrix.T, k=k, dim=1)

            # Calculate recall@k
            correct_predictions = 0
            for i in range(n_samples):
                if i in top_k_indices[i]:  # Check if correct radar is in top-k
                    correct_predictions += 1

            recall_k = correct_predictions / n_samples
            results[f'text_to_radar_recall@{k}'] = recall_k
            print(f"  Recall@{k}: {recall_k:.4f}")

        # Calculate Mean Reciprocal Rank (MRR)
        print("\n3. Mean Reciprocal Rank (MRR):")

        # Radar→Text MRR
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
        radar_to_text_ranks = []
        for i in range(n_samples):
            rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1
            radar_to_text_ranks.append(1.0 / rank)

        radar_to_text_mrr = np.mean(radar_to_text_ranks)
        results['radar_to_text_mrr'] = radar_to_text_mrr
        print(f"  Radar→Text MRR: {radar_to_text_mrr:.4f}")

        # Text→Radar MRR
        _, sorted_indices = torch.sort(similarity_matrix.T, dim=1, descending=True)
        text_to_radar_ranks = []
        for i in range(n_samples):
            rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1
            text_to_radar_ranks.append(1.0 / rank)

        text_to_radar_mrr = np.mean(text_to_radar_ranks)
        results['text_to_radar_mrr'] = text_to_radar_mrr
        print(f"  Text→Radar MRR: {text_to_radar_mrr:.4f}")

        # Median Rank
        print("\n4. Median Rank:")
        radar_to_text_median_rank = np.median([int(1.0/r) for r in radar_to_text_ranks])
        text_to_radar_median_rank = np.median([int(1.0/r) for r in text_to_radar_ranks])

        results['radar_to_text_median_rank'] = radar_to_text_median_rank
        results['text_to_radar_median_rank'] = text_to_radar_median_rank

        print(f"  Radar→Text Median Rank: {radar_to_text_median_rank}")
        print(f"  Text→Radar Median Rank: {text_to_radar_median_rank}")

        return results, similarity_matrix

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
        """Run complete evaluation"""
        print("=" * 60)
        print("CLIP Model Evaluation")
        print("=" * 60)

        # Extract features
        radar_features, text_features, captions = self.extract_features()

        # Evaluate retrieval
        results, similarity_matrix = self.evaluate_retrieval(radar_features, text_features)

        # Analyze similarity distribution
        self.analyze_similarity_distribution(similarity_matrix)

        # Evaluate classification accuracy
        acc_r2t, acc_t2r = self.evaluate_classification_accuracy(similarity_matrix)
        results['radar_to_text_accuracy'] = acc_r2t
        results['text_to_radar_accuracy'] = acc_t2r

        # Save results
        if output_path:
            self.save_results(results, output_path)

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIP model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to model configuration file')
    parser.add_argument('--output_path', type=str, default='clip_evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')

    args = parser.parse_args()

    # Create evaluator
    evaluator = CLIPEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )

    # Run evaluation
    results = evaluator.run_evaluation(output_path=args.output_path)

    print(f"\n📊 Final Results Summary:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()