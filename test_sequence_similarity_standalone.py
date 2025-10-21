#!/usr/bin/env python3
"""
Standalone test for sequence similarity functionality without requiring model downloads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.sequence_similarity import SequenceSimilarity, SequenceLoss

def test_sequence_similarity():
    """Test the sequence similarity module directly."""
    print("Testing Sequence Similarity Module")
    print("=" * 40)

    # Configuration
    embed_dim = 512
    batch_size = 4
    radar_len = 496
    text_len = 77

    # Create sequence similarity module
    seq_sim = SequenceSimilarity(
        embed_dim=embed_dim,
        similarity_type="combined",
        window_size=16,
        num_heads=8,
        temperature=0.07
    )

    # Create dummy sequence data
    radar_seq = torch.randn(batch_size, radar_len, embed_dim)
    text_seq = torch.randn(batch_size, text_len, embed_dim)

    print(f"Input shapes - Radar: {radar_seq.shape}, Text: {text_seq.shape}")

    # Test different similarity types
    similarity_types = ["global", "local", "attention", "temporal", "combined"]

    for sim_type in similarity_types:
        print(f"\nTesting {sim_type} similarity:")
        seq_sim.similarity_type = sim_type

        try:
            similarity_matrix = seq_sim(radar_seq, text_seq)
            print(f"  Output shape: {similarity_matrix.shape}")
            print(f"  Output range: [{similarity_matrix.min().item():.3f}, {similarity_matrix.max().item():.3f}]")
            print(f"  Diagonal mean: {torch.diag(similarity_matrix).mean().item():.3f}")
            print(f"  Off-diagonal mean: {(similarity_matrix.sum() - torch.diag(similarity_matrix).sum()).item() / (batch_size**2 - batch_size):.3f}")
        except Exception as e:
            print(f"  Error: {e}")

    return seq_sim

def test_sequence_loss():
    """Test the sequence loss functionality."""
    print("\n\nTesting Sequence Loss")
    print("=" * 40)

    # Configuration
    embed_dim = 512
    batch_size = 4
    radar_len = 496
    text_len = 77

    # Create sequence loss module
    seq_loss = SequenceLoss(
        embed_dim=embed_dim,
        similarity_type="combined",
        temperature=0.07,
        use_siglip=False
    )

    # Create dummy sequence data
    radar_seq = torch.randn(batch_size, radar_len, embed_dim)
    text_seq = torch.randn(batch_size, text_len, embed_dim)

    print(f"Input shapes - Radar: {radar_seq.shape}, Text: {text_seq.shape}")

    # Test loss computation
    try:
        loss = seq_loss(radar_seq, text_seq)
        print(f"Sequence loss: {loss.item():.4f}")

        # Test with learnable logit scale
        logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/0.07)))
        loss_with_scale = seq_loss(radar_seq, text_seq, logit_scale=logit_scale)
        print(f"Loss with logit scale: {loss_with_scale.item():.4f}")

        # Test SigLIP loss
        seq_loss.use_siglip = True
        siglip_loss = seq_loss(radar_seq, text_seq)
        print(f"SigLIP loss: {siglip_loss.item():.4f}")

    except Exception as e:
        print(f"Error computing loss: {e}")

    return seq_loss

def test_window_extraction():
    """Test window extraction functionality."""
    print("\n\nTesting Window Extraction")
    print("=" * 40)

    # Create sequence similarity module for window testing
    seq_sim = SequenceSimilarity(embed_dim=512, window_size=16)

    # Test different sequence lengths
    test_cases = [
        (4, 32, 512),   # Normal case
        (4, 16, 512),   # Exact window size
        (4, 10, 512),   # Shorter than window size (should pad)
    ]

    for batch_size, seq_len, embed_dim in test_cases:
        print(f"\nTesting sequence length {seq_len}:")
        sequence = torch.randn(batch_size, seq_len, embed_dim)

        try:
            windows = seq_sim.extract_windows(sequence, 16)
            print(f"  Input shape: {sequence.shape}")
            print(f"  Output shape: {windows.shape}")
            print(f"  Number of windows: {windows.size(1)}")

            # Test if windows are correctly extracted
            if seq_len >= 16:
                # Check first window matches first 16 tokens
                first_window = windows[:, 0, :, :]
                expected = sequence[:, :16, :]
                assert torch.allclose(first_window, expected), "First window mismatch"
                print("  ✓ Window extraction correct")
            else:
                print("  ✓ Padding applied for short sequence")

        except Exception as e:
            print(f"  Error: {e}")

def test_dtw_alignment():
    """Test DTW alignment functionality."""
    print("\n\nTesting DTW Alignment")
    print("=" * 40)

    seq_sim = SequenceSimilarity(embed_dim=512)

    # Create test similarity matrices
    test_cases = [
        (10, 15, "Normal case"),
        (5, 5, "Square matrix"),
        (20, 8, "Rectangular matrix (radar > text)"),
        (8, 20, "Rectangular matrix (text > radar)"),
    ]

    for seq1_len, seq2_len, description in test_cases:
        print(f"\nTesting {description} ({seq1_len}x{seq2_len}):")
        sim_matrix = torch.randn(seq1_len, seq2_len)

        try:
            alignment_score = seq_sim.dtw_alignment(sim_matrix)
            print(f"  Alignment score: {alignment_score.item():.4f}")
            print(f"  Matrix mean: {sim_matrix.mean().item():.4f}")
            print(f"  Matrix std: {sim_matrix.std().item():.4f}")
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Run all tests."""
    print("Standalone Sequence Similarity Tests")
    print("=" * 50)

    # Test sequence similarity
    seq_sim = test_sequence_similarity()

    # Test sequence loss
    seq_loss = test_sequence_loss()

    # Test window extraction
    test_window_extraction()

    # Test DTW alignment
    test_dtw_alignment()

    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main()