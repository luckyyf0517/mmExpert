#!/usr/bin/env python3
"""
Quick CLIP Evaluation Script

Simple evaluation for quick testing during development
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append('/root/autodl-tmp/mmExpert')

from src.model.clip import CLIP
from src.data_interface import HumanDInterface
from easydict import EasyDict as edict
import json

def quick_evaluation(model_path, config_path, num_batches=10):
    """Quick evaluation on a subset of test data"""

    print(f"Quick Evaluation: {model_path}")
    print(f"Using {num_batches} batches from test set")

    # Load model
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_cfg = config['model_cfg']['params']
    model = CLIP(**model_cfg)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load test data
    data_cfg = config['data_cfg']
    data_interface = HumanDInterface(data_cfg['params'])
    data_interface.setup('test')
    test_dataloader = data_interface.test_dataloader()

    print(f"Test dataset size: {len(test_dataloader.dataset)}")

    # Extract features from subset
    all_radar_features = []
    all_text_features = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Extracting features")):
            if i >= num_batches:
                break

            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Get features
            radar_features = model.encode_radar(
                batch['input_wave_range'],
                batch['input_wave_doppler'],
                batch['input_wave_azimuth']
            )

            text_features = model.encode_text(batch['caption'], device=device)

            # Normalize
            radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            all_radar_features.append(radar_features.cpu())
            all_text_features.append(text_features.cpu())

    # Concatenate features
    radar_features = torch.cat(all_radar_features, dim=0).to(device)
    text_features = torch.cat(all_text_features, dim=0).to(device)

    print(f"Feature shapes: Radar {radar_features.shape}, Text {text_features.shape}")

    # Compute similarity matrix
    similarity_matrix = torch.matmul(radar_features, text_features.T)

    # Simple accuracy evaluation (1-to-1 matching)
    n_samples = len(radar_features)

    # Radar -> Text
    _, predicted_texts = torch.max(similarity_matrix, dim=1)
    correct_indices = torch.arange(n_samples, device=device)
    r2t_accuracy = (predicted_texts == correct_indices).float().mean().item()

    # Text -> Radar
    _, predicted_radars = torch.max(similarity_matrix.T, dim=1)
    t2r_accuracy = (predicted_radars == correct_indices).float().mean().item()

    # Recall@K
    k_values = [1, 5, 10]
    recall_results = {}

    for k in k_values:
        # Radar -> Text Recall@K
        _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
        r2t_recall = sum([1 for i in range(n_samples) if i in top_k_indices[i]]) / n_samples

        # Text -> Radar Recall@K
        _, top_k_indices = torch.topk(similarity_matrix.T, k=k, dim=1)
        t2r_recall = sum([1 for i in range(n_samples) if i in top_k_indices[i]]) / n_samples

        recall_results[f'R2T_Recall@{k}'] = r2t_recall
        recall_results[f'T2R_Recall@{k}'] = t2r_recall

    # Print results
    print("\n" + "="*50)
    print("QUICK EVALUATION RESULTS")
    print("="*50)
    print(f"Samples evaluated: {n_samples}")
    print(f"\nClassification Accuracy:")
    print(f"  Radar→Text: {r2t_accuracy:.4f}")
    print(f"  Text→Radar: {t2r_accuracy:.4f}")

    print(f"\nRecall@K:")
    for k in k_values:
        print(f"  Recall@{k}: R2T={recall_results[f'R2T_Recall@{k}']:.4f}, "
              f"T2R={recall_results[f'T2R_Recall@{k}']:.4f}")

    # Similarity analysis
    diagonal_scores = torch.diag(similarity_matrix).cpu().numpy()
    off_diagonal_scores = similarity_matrix[torch.arange(n_samples) != torch.arange(n_samples).unsqueeze(1)].cpu().numpy()

    print(f"\nSimilarity Analysis:")
    print(f"  Correct pairs (mean): {np.mean(diagonal_scores):.4f}")
    print(f"  Wrong pairs (mean):   {np.mean(off_diagonal_scores):.4f}")
    print(f"  Separation:           {np.mean(diagonal_scores) - np.mean(off_diagonal_scores):.4f}")

    print("="*50)

    return {
        'samples': n_samples,
        'r2t_accuracy': r2t_accuracy,
        't2r_accuracy': t2r_accuracy,
        **recall_results,
        'correct_similarity': np.mean(diagonal_scores),
        'wrong_similarity': np.mean(off_diagonal_scores),
        'separation': np.mean(diagonal_scores) - np.mean(off_diagonal_scores)
    }

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/model.ckpt"
    config_path = "config/clip.yaml"

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        config_path = sys.argv[2]

    try:
        results = quick_evaluation(model_path, config_path, num_batches=20)
        print(f"\n✅ Quick evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()