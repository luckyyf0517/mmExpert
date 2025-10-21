#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced CLIP model with sequence similarity.

This script shows how to:
1. Configure CLIP with sequence similarity
2. Train with sequence-level losses
3. Extract sequence features for analysis
"""

import torch
import torch.nn as nn
from src.model.clip_model import CLIP

def create_clip_with_sequence_similarity():
    """
    Create a CLIP model with sequence similarity enabled.
    """
    # Model configuration
    encoder_cfg = {
        'model_name': 'vit_base_patch16_224',
        'embed_dim': 512,
        'range_resolution': (256, 496),
        'doppler_resolution': (128, 496),
        'azimuth_resolution': (128, 496),
        'pretrained': False,
        'fusion_method': 'concat',
        'adaptive_patch_size': True,
        'radar_views': 'all'
    }

    text_cfg = {
        'model_name': 'bert-base-uncased',
        'text_pooling': 'pooler',
        'unfreeze_last_layer_num': 2
    }

    # Create CLIP model with sequence similarity
    model = CLIP(
        encoder_cfg=encoder_cfg,
        text_cfg=text_cfg,
        context_length=512,
        transformer_width=512,
        transformer_layers=6,
        transformer_heads=8,
        temperature=0.07,
        use_siglip=False,  # Set to True to use SigLIP loss
        learning_rate=1e-4,
        max_epochs=50,
        # Sequence similarity configuration
        use_sequence_similarity=True,
        sequence_similarity_type="combined",  # Options: "global", "local", "attention", "temporal", "combined"
        sequence_similarity_weight=0.5  # Weight for sequence loss relative to standard CLIP loss
    )

    return model

def example_training_step():
    """
    Example showing how the enhanced training step works.
    """
    model = create_clip_with_sequence_similarity()
    model.train()

    # Create dummy data
    batch_size = 4
    time_frames = 496

    # Radar data
    radar_data = {
        'range_time': torch.randn(batch_size, 256, time_frames),
        'doppler_time': torch.randn(batch_size, 128, time_frames),
        'azimuth_time': torch.randn(batch_size, 128, time_frames)
    }

    # Text data
    text = [
        "A person walking forward",
        "A car moving to the left",
        "A stationary object",
        "Multiple objects approaching"
    ]

    # Create batch dict (as would be provided by dataloader)
    batch = {
        'input_wave_range': radar_data['range_time'],
        'input_wave_doppler': radar_data['doppler_time'],
        'input_wave_azimuth': radar_data['azimuth_time'],
        'caption': text
    }

    # Forward pass with sequence features
    radar_pooled, text_pooled, radar_seq, text_seq = model.forward(radar_data, text, return_sequences=True)

    print("Sequence feature shapes:")
    print(f"Radar pooled: {radar_pooled.shape}")
    print(f"Text pooled: {text_pooled.shape}")
    print(f"Radar sequence: {radar_seq.shape}")
    print(f"Text sequence: {text_seq.shape}")

    # Training step (this will compute both standard and sequence losses)
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item()}")

    return model

def example_sequence_analysis():
    """
    Example showing how to extract and analyze sequence features.
    """
    model = create_clip_with_sequence_similarity()
    model.eval()

    # Create dummy data
    batch_size = 2
    time_frames = 496

    radar_data = {
        'range_time': torch.randn(batch_size, 256, time_frames),
        'doppler_time': torch.randn(batch_size, 128, time_frames),
        'azimuth_time': torch.randn(batch_size, 128, time_frames)
    }

    text = ["Person walking", "Car driving"]

    # Extract sequence features
    radar_seq, text_seq = model.encode_sequences(radar_data, text)

    print("\nSequence analysis:")
    print(f"Radar sequence length: {radar_seq.size(1)}")
    print(f"Text sequence length: {text_seq.size(1)}")

    # Analyze temporal dynamics
    radar_std = torch.std(radar_seq, dim=1)  # Temporal variation
    text_std = torch.std(text_seq, dim=1)   # Temporal variation

    print(f"Radar temporal variation: {radar_std.mean().item():.4f}")
    print(f"Text temporal variation: {text_std.mean().item():.4f}")

    # Compute sequence similarity matrix
    if hasattr(model, 'sequence_loss_fn'):
        similarity_fn = model.sequence_loss_fn.similarity_fn
        similarity_matrix = similarity_fn(radar_seq, text_seq)
        print(f"Sequence similarity matrix:\n{similarity_matrix}")

def example_configuration_options():
    """
    Demonstrate different configuration options for sequence similarity.
    """

    print("\nConfiguration Options:")

    # 1. Global similarity only (most computationally efficient)
    print("\n1. Global similarity only:")
    model_global = create_clip_with_sequence_similarity()
    model_global.sequence_loss_fn.similarity_fn.similarity_type = "global"
    print("   - Computationally efficient")
    print("   - Uses sequence pooling")

    # 2. Local/window-based similarity
    print("\n2. Local/window-based similarity:")
    model_local = create_clip_with_sequence_similarity()
    model_local.sequence_loss_fn.similarity_fn.similarity_type = "local"
    model_local.sequence_loss_fn.similarity_fn.window_size = 32
    print("   - Captures local patterns")
    print("   - Uses sliding windows")

    # 3. Attention-based similarity
    print("\n3. Attention-based similarity:")
    model_attention = create_clip_with_sequence_similarity()
    model_attention.sequence_loss_fn.similarity_fn.similarity_type = "attention"
    print("   - Learns cross-modal alignment")
    print("   - Uses attention mechanisms")

    # 4. Temporal alignment similarity
    print("\n4. Temporal alignment similarity:")
    model_temporal = create_clip_with_sequence_similarity()
    model_temporal.sequence_loss_fn.similarity_fn.similarity_type = "temporal"
    print("   - Aligns sequences in time")
    print("   - Uses DTW-like alignment")

    # 5. Combined similarity (default)
    print("\n5. Combined similarity (recommended):")
    print("   - Combines all similarity types")
    print("   - Most comprehensive approach")
    print("   - Weights can be adjusted via sequence_similarity_fn.weights")

def main():
    """
    Main function demonstrating sequence similarity features.
    """
    print("CLIP with Sequence Similarity - Example Usage")
    print("=" * 50)

    # Example 1: Training step
    print("\n1. Training Step Example:")
    example_training_step()

    # Example 2: Sequence analysis
    print("\n2. Sequence Analysis Example:")
    example_sequence_analysis()

    # Example 3: Configuration options
    print("\n3. Configuration Options:")
    example_configuration_options()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")

if __name__ == "__main__":
    main()