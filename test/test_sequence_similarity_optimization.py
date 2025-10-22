#!/usr/bin/env python3
"""
Test script to validate the optimization of SequenceSimilarity local_similarity method.
This script compares the performance and correctness of the original vs optimized implementations.
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import Tuple
import sys
import os

# Add project root to path
sys.path.append('/root/autodl-tmp/mmExpert')

from src.model.sequence_similarity import SequenceSimilarity


class OriginalSequenceSimilarity:
    """
    Original implementation with double loops for comparison.
    """

    def __init__(self, embed_dim: int, window_size: int = 16, temperature: float = 0.07):
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.temperature = temperature
        self.local_proj = torch.nn.Linear(embed_dim, embed_dim)

    def extract_windows(self, sequence, window_size):
        """Extract sliding windows from sequence."""
        batch_size, seq_len, embed_dim = sequence.size()

        if seq_len < window_size:
            padding = window_size - seq_len
            sequence = F.pad(sequence, (0, 0, 0, padding))
            seq_len = window_size

        num_windows = seq_len - window_size + 1
        windows = []

        for i in range(num_windows):
            windows.append(sequence[:, i:i+window_size, :])

        windows = torch.stack(windows, dim=1)
        return windows

    def local_similarity_original(self, radar_seq, text_seq):
        """Original double-loop implementation."""
        batch_size = radar_seq.size(0)

        # Extract sliding windows
        radar_windows = self.extract_windows(radar_seq, self.window_size)
        text_windows = self.extract_windows(text_seq, self.window_size)

        # Pool windows
        radar_pooled = radar_windows.mean(dim=2)
        text_pooled = text_windows.mean(dim=2)

        # Apply projection
        radar_pooled = self.local_proj(radar_pooled)
        text_pooled = self.local_proj(text_pooled)

        # Normalize
        radar_pooled = F.normalize(radar_pooled, dim=-1)
        text_pooled = F.normalize(text_pooled, dim=-1)

        # Original double loop implementation
        local_similarities = []

        for i in range(batch_size):
            batch_similarities = []
            for j in range(batch_size):
                radar_windows = radar_pooled[i]
                text_windows = text_pooled[j]

                # Compute similarity matrix between windows
                window_sim_matrix = torch.matmul(radar_windows, text_windows.T) / self.temperature

                # Pool across windows to get sequence-level similarity
                similarity_score = window_sim_matrix.max()
                batch_similarities.append(similarity_score)

            local_similarities.append(torch.stack(batch_similarities))

        similarity = torch.stack(local_similarities)
        return similarity


def create_test_data(batch_size: int, radar_len: int, text_len: int, embed_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic test data."""
    torch.manual_seed(42)  # For reproducible results

    radar_seq = torch.randn(batch_size, radar_len, embed_dim)
    text_seq = torch.randn(batch_size, text_len, embed_dim)

    return radar_seq, text_seq


def test_correctness():
    """Test if optimized implementation produces the same results as original."""
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    # Test parameters
    batch_size = 8
    radar_len = 100
    text_len = 77
    embed_dim = 256
    window_size = 16
    temperature = 0.07

    # Create test data
    radar_seq, text_seq = create_test_data(batch_size, radar_len, text_len, embed_dim)

    # Initialize both implementations
    original_sim = OriginalSequenceSimilarity(embed_dim, window_size, temperature)
    optimized_sim = SequenceSimilarity(
        embed_dim=embed_dim,
        similarity_type="local",
        window_size=window_size,
        temperature=temperature
    )

    # Copy projection weights to ensure fair comparison
    with torch.no_grad():
        optimized_sim.local_proj.weight.copy_(original_sim.local_proj.weight)
        optimized_sim.local_proj.bias.copy_(original_sim.local_proj.bias)

    # Compute similarities
    with torch.no_grad():
        original_result = original_sim.local_similarity_original(radar_seq, text_seq)
        optimized_result = optimized_sim.local_similarity(radar_seq, text_seq)

    # Compare results
    max_diff = torch.max(torch.abs(original_result - optimized_result)).item()
    mean_diff = torch.mean(torch.abs(original_result - optimized_result)).item()

    print(f"Batch size: {batch_size}")
    print(f"Radar seq length: {radar_len}, Text seq length: {text_len}")
    print(f"Window size: {window_size}, Embed dim: {embed_dim}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("✅ CORRECTNESS TEST PASSED: Results are identical within tolerance")
    else:
        print("❌ CORRECTNESS TEST FAILED: Results differ significantly")
        print(f"Original result shape: {original_result.shape}")
        print(f"Optimized result shape: {optimized_result.shape}")
        print(f"Original sample:\n{original_result[:2, :2]}")
        print(f"Optimized sample:\n{optimized_result[:2, :2]}")

    return max_diff < 1e-4


def test_performance():
    """Test performance improvement of optimized implementation."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)

    # Test different batch sizes
    batch_sizes = [4, 8, 16, 32]
    radar_len = 100
    text_len = 77
    embed_dim = 256
    window_size = 16
    temperature = 0.07
    num_runs = 5  # Number of runs for averaging

    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")

        # Create test data
        radar_seq, text_seq = create_test_data(batch_size, radar_len, text_len, embed_dim)

        # Initialize implementations
        original_sim = OriginalSequenceSimilarity(embed_dim, window_size, temperature)
        optimized_sim = SequenceSimilarity(
            embed_dim=embed_dim,
            similarity_type="local",
            window_size=window_size,
            temperature=temperature
        )

        # Copy weights
        with torch.no_grad():
            optimized_sim.local_proj.weight.copy_(original_sim.local_proj.weight)
            optimized_sim.local_proj.bias.copy_(original_sim.local_proj.bias)

        # Warm up
        with torch.no_grad():
            original_sim.local_similarity_original(radar_seq, text_seq)
            optimized_sim.local_similarity(radar_seq, text_seq)

        # Test original implementation
        original_times = []
        for _ in range(num_runs):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()
            with torch.no_grad():
                original_result = original_sim.local_similarity_original(radar_seq, text_seq)
            end_time = time.time()
            original_times.append(end_time - start_time)

        # Test optimized implementation
        optimized_times = []
        for _ in range(num_runs):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_time = time.time()
            with torch.no_grad():
                optimized_result = optimized_sim.local_similarity(radar_seq, text_seq)
            end_time = time.time()
            optimized_times.append(end_time - start_time)

        # Calculate statistics
        original_mean = np.mean(original_times)
        original_std = np.std(original_times)
        optimized_mean = np.mean(optimized_times)
        optimized_std = np.std(optimized_times)
        speedup = original_mean / optimized_mean

        print(f"  Original:   {original_mean:.4f}s ± {original_std:.4f}s")
        print(f"  Optimized:  {optimized_mean:.4f}s ± {optimized_std:.4f}s")
        print(f"  Speedup:    {speedup:.2f}x")

        results.append({
            'batch_size': batch_size,
            'original_time': original_mean,
            'optimized_time': optimized_mean,
            'speedup': speedup
        })

    return results


def test_memory_usage():
    """Test memory usage comparison."""
    print("\n" + "=" * 60)
    print("MEMORY USAGE TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return

    batch_size = 16
    radar_len = 200
    text_len = 150
    embed_dim = 512
    window_size = 16

    # Create test data
    radar_seq = torch.randn(batch_size, radar_len, embed_dim).cuda()
    text_seq = torch.randn(batch_size, text_len, embed_dim).cuda()

    optimized_sim = SequenceSimilarity(
        embed_dim=embed_dim,
        similarity_type="local",
        window_size=window_size,
        temperature=0.07
    ).cuda()

    # Test different chunk sizes
    chunk_sizes = [4, 8, 16, 32]

    for chunk_size in chunk_sizes:
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            similarity = optimized_sim._local_similarity_optimized(
                optimized_sim.extract_windows(radar_seq, window_size).mean(dim=2),
                optimized_sim.extract_windows(text_seq, window_size).mean(dim=2),
                chunk_size=chunk_size
            )

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Chunk size {chunk_size:2d}: Peak memory usage: {peak_memory:.1f} MB")

    # Clean up
    del radar_seq, text_seq, optimized_sim
    torch.cuda.empty_cache()


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("EDGE CASES TEST")
    print("=" * 60)

    embed_dim = 256
    window_size = 16
    temperature = 0.07

    sim = SequenceSimilarity(
        embed_dim=embed_dim,
        similarity_type="local",
        window_size=window_size,
        temperature=temperature
    )

    test_cases = [
        # (batch_size, radar_len, text_len, description)
        (1, 10, 8, "Single sample, short sequences"),
        (2, window_size-1, window_size-1, "Sequences shorter than window"),
        (4, 50, 50, "Equal length sequences"),
        (8, 200, 77, "Long radar sequence"),
    ]

    for batch_size, radar_len, text_len, description in test_cases:
        try:
            radar_seq = torch.randn(batch_size, radar_len, embed_dim)
            text_seq = torch.randn(batch_size, text_len, embed_dim)

            with torch.no_grad():
                result = sim.local_similarity(radar_seq, text_seq)

            expected_shape = (batch_size, batch_size)
            if result.shape == expected_shape:
                print(f"✅ {description}: PASS (shape {result.shape})")
            else:
                print(f"❌ {description}: FAIL (expected {expected_shape}, got {result.shape})")

        except Exception as e:
            print(f"❌ {description}: ERROR ({str(e)})")


def main():
    """Run all tests."""
    print("Sequence Similarity Optimization Test Suite")
    print("Testing optimized local_similarity implementation")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Run tests
    correctness_passed = test_correctness()
    performance_results = test_performance()
    test_memory_usage()
    test_edge_cases()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if correctness_passed:
        print("✅ All correctness tests passed")
    else:
        print("❌ Some correctness tests failed")

    print("\nPerformance Results:")
    for result in performance_results:
        print(f"  Batch {result['batch_size']:2d}: {result['speedup']:.2f}x speedup")

    avg_speedup = np.mean([r['speedup'] for r in performance_results])
    print(f"Average speedup: {avg_speedup:.2f}x")

    print("\nOptimization successful! 🚀")


if __name__ == "__main__":
    main()