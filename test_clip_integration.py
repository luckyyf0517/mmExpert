#!/usr/bin/env python3
"""
Test CLIP integration with sequence similarity using mock encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.sequence_similarity import SequenceSimilarity, SequenceLoss


class MockRadarEncoder(nn.Module):
    """Mock radar encoder that returns sequence features."""
    def __init__(self, embed_dim=512, output_seq_len=496):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_seq_len = output_seq_len

        # Simple CNN + LSTM to simulate radar processing
        self.conv = nn.Conv1d(256, embed_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, radar_data):
        """
        Args:
            radar_data: dict with range_time, doppler_time, azimuth_time
        Returns:
            pooled: [b, embed_dim]
            sequence: [b, seq_len, embed_dim]
        """
        batch_size = radar_data['range_time'].size(0)

        # Simulate processing by taking one modality
        x = radar_data['range_time']  # [b, 256, 496]

        # Process with conv
        x = self.conv(x)  # [b, embed_dim, 496]
        x = x.transpose(1, 2)  # [b, 496, embed_dim]

        # Process with LSTM
        x, _ = self.lstm(x)  # [b, 496, embed_dim]

        # Project
        x = self.proj(x)

        # Pooled representation (EOT equivalent)
        pooled = x.mean(dim=1)  # [b, embed_dim]

        # Normalize
        pooled = F.normalize(pooled, dim=-1)

        return pooled, x


class MockTextEncoder(nn.Module):
    """Mock text encoder that returns sequence features."""
    def __init__(self, embed_dim=512, vocab_size=30522, max_seq_len=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=6
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text):
        """
        Args:
            text: list of strings
        Returns:
            pooled: [b, embed_dim]
            sequence: [b, seq_len, embed_dim]
        """
        batch_size = len(text)
        seq_len = min(32, self.max_seq_len)  # Fixed length for testing

        # Mock tokenization (random tokens for testing)
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Embed tokens
        x = self.token_embedding(token_ids)  # [b, seq_len, embed_dim]

        # Add positional encoding
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(pos_ids)

        # Process with transformer
        x = self.transformer(x)  # [b, seq_len, embed_dim]

        # Project
        x = self.proj(x)

        # Pooled representation (using last token as EOT)
        pooled = x[:, -1, :]  # [b, embed_dim]

        # Normalize
        pooled = F.normalize(pooled, dim=-1)

        return pooled, x


class MockCLIPWithSequence(nn.Module):
    """Mock CLIP model with sequence similarity support."""
    def __init__(self, embed_dim=512, use_sequence_similarity=True,
                 sequence_similarity_type="combined", sequence_similarity_weight=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_sequence_similarity = use_sequence_similarity
        self.sequence_similarity_weight = sequence_similarity_weight

        # Encoders
        self.radar_encoder = MockRadarEncoder(embed_dim)
        self.text_encoder = MockTextEncoder(embed_dim)

        # Sequence similarity
        if use_sequence_similarity:
            self.sequence_loss_fn = SequenceLoss(
                embed_dim=embed_dim,
                similarity_type=sequence_similarity_type,
                temperature=0.07
            )

        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def forward(self, radar_data, text, return_sequences=False):
        """Forward pass."""
        # Get features
        radar_pooled, radar_seq = self.radar_encoder(radar_data)
        text_pooled, text_seq = self.text_encoder(text)

        if return_sequences:
            return radar_pooled, text_pooled, radar_seq, text_seq

        return radar_pooled, text_pooled

    def compute_contrastive_loss(self, radar_features, text_features):
        """Compute standard contrastive loss."""
        # Compute similarity matrix
        logits = torch.matmul(radar_features, text_features.T) * self.logit_scale.exp()

        # Labels are diagonal (each radar matches its text)
        labels = torch.arange(radar_features.size(0), device=radar_features.device)

        # Compute loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

    def training_step(self, batch, batch_idx):
        """Training step with sequence similarity."""
        # Extract data
        radar_data = {
            'range_time': batch['input_wave_range'],
            'doppler_time': batch['input_wave_doppler'],
            'azimuth_time': batch['input_wave_azimuth']
        }
        text = batch['caption']

        # Get features
        radar_pooled, text_pooled, radar_seq, text_seq = self.forward(
            radar_data, text, return_sequences=True
        )

        # Standard CLIP loss
        loss_clip = self.compute_contrastive_loss(radar_pooled, text_pooled)

        if self.use_sequence_similarity:
            # Sequence similarity loss
            loss_seq = self.sequence_loss_fn(radar_seq, text_seq)

            # Combined loss
            total_loss = (1 - self.sequence_similarity_weight) * loss_clip + \
                        self.sequence_similarity_weight * loss_seq

            return {
                'loss_total': total_loss,
                'loss_clip': loss_clip,
                'loss_seq': loss_seq
            }
        else:
            return {
                'loss_total': loss_clip,
                'loss_clip': loss_clip
            }


def test_clip_integration():
    """Test CLIP model with sequence similarity integration."""
    print("Testing CLIP Integration with Sequence Similarity")
    print("=" * 50)

    # Create model
    model = MockCLIPWithSequence(
        embed_dim=512,
        use_sequence_similarity=True,
        sequence_similarity_type="combined",
        sequence_similarity_weight=0.5
    )

    model.train()

    # Create dummy data
    batch_size = 4
    time_frames = 496

    batch = {
        'input_wave_range': torch.randn(batch_size, 256, time_frames),
        'input_wave_doppler': torch.randn(batch_size, 128, time_frames),
        'input_wave_azimuth': torch.randn(batch_size, 128, time_frames),
        'caption': [
            "A person walking forward",
            "A car moving to the left",
            "A stationary object",
            "Multiple objects approaching"
        ]
    }

    print(f"Batch size: {batch_size}")
    print(f"Radar data shapes: {batch['input_wave_range'].shape}")

    # Prepare radar data
    radar_data = {
        'range_time': batch['input_wave_range'],
        'doppler_time': batch['input_wave_doppler'],
        'azimuth_time': batch['input_wave_azimuth']
    }

    # Test forward pass
    radar_pooled, text_pooled, radar_seq, text_seq = model.forward(
        radar_data,
        batch['caption'],
        return_sequences=True
    )

    print(f"Radar pooled: {radar_pooled.shape}")
    print(f"Text pooled: {text_pooled.shape}")
    print(f"Radar sequence: {radar_seq.shape}")
    print(f"Text sequence: {text_seq.shape}")

    # Test training step
    losses = model.training_step(batch, 0)

    print(f"\nLoss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value}")

    return model


def test_different_similarity_types():
    """Test different similarity types."""
    print("\n\nTesting Different Similarity Types")
    print("=" * 50)

    similarity_types = ["global", "local", "attention", "temporal", "combined"]

    # Create test data
    batch_size = 2
    batch = {
        'input_wave_range': torch.randn(batch_size, 256, 496),
        'input_wave_doppler': torch.randn(batch_size, 128, 496),
        'input_wave_azimuth': torch.randn(batch_size, 128, 496),
        'caption': ["Person walking", "Car driving"]
    }

    results = {}

    for sim_type in similarity_types:
        print(f"\nTesting {sim_type} similarity:")

        # Create model with specific similarity type
        model = MockCLIPWithSequence(
            embed_dim=512,
            use_sequence_similarity=True,
            sequence_similarity_type=sim_type,
            sequence_similarity_weight=0.5
        )

        try:
            # Test training step
            losses = model.training_step(batch, 0)

            total_loss = losses['loss_total'].item()
            seq_loss = losses.get('loss_seq', 0)

            if isinstance(seq_loss, torch.Tensor):
                seq_loss = seq_loss.item()

            print(f"  Total loss: {total_loss:.4f}")
            if seq_loss > 0:
                print(f"  Sequence loss: {seq_loss:.4f}")

            results[sim_type] = {'total': total_loss, 'seq': seq_loss}

        except Exception as e:
            print(f"  Error: {e}")
            results[sim_type] = {'error': str(e)}

    return results


def test_sequence_analysis():
    """Test sequence feature analysis."""
    print("\n\nTesting Sequence Feature Analysis")
    print("=" * 50)

    # Create model
    model = MockCLIPWithSequence(embed_dim=512)
    model.eval()

    # Create test data
    batch_size = 2
    batch = {
        'input_wave_range': torch.randn(batch_size, 256, 496),
        'input_wave_doppler': torch.randn(batch_size, 128, 496),
        'input_wave_azimuth': torch.randn(batch_size, 128, 496),
        'caption': ["Person walking", "Car driving"]
    }

    # Prepare radar data
    radar_data = {
        'range_time': batch['input_wave_range'],
        'doppler_time': batch['input_wave_doppler'],
        'azimuth_time': batch['input_wave_azimuth']
    }

    # Extract sequence features
    radar_pooled, text_pooled, radar_seq, text_seq = model.forward(
        radar_data,
        batch['caption'],
        return_sequences=True
    )

    # Analyze sequences
    print(f"Sequence lengths:")
    print(f"  Radar: {radar_seq.size(1)}")
    print(f"  Text: {text_seq.size(1)}")

    # Temporal variation
    radar_std = torch.std(radar_seq, dim=1)  # [b, embed_dim]
    text_std = torch.std(text_seq, dim=1)   # [b, embed_dim]

    print(f"\nTemporal variation (std across time):")
    print(f"  Radar: {radar_std.mean().item():.4f} ± {radar_std.std().item():.4f}")
    print(f"  Text:  {text_std.mean().item():.4f} ± {text_std.std().item():.4f}")

    # Feature norms across time
    radar_norms = torch.norm(radar_seq, dim=-1)  # [b, seq_len]
    text_norms = torch.norm(text_seq, dim=-1)    # [b, seq_len]

    print(f"\nFeature norms:")
    print(f"  Radar: {radar_norms.mean().item():.4f} ± {radar_norms.std().item():.4f}")
    print(f"  Text:  {text_norms.mean().item():.4f} ± {text_norms.std().item():.4f}")

    # Cross-modal similarity
    similarity = torch.matmul(radar_seq, text_seq.transpose(-2, -1))  # [b, seq_radar, seq_text]

    print(f"\nCross-modal sequence similarity:")
    print(f"  Mean: {similarity.mean().item():.4f}")
    print(f"  Max: {similarity.max().item():.4f}")
    print(f"  Min: {similarity.min().item():.4f}")

    # Sequence alignment (find best matching positions)
    best_matches = similarity.argmax(dim=-1)  # [b, seq_radar]

    print(f"\nBest alignment positions (radar->text):")
    print(f"  Sample 0: {best_matches[0][:10].tolist()}...")  # First 10 positions
    print(f"  Sample 1: {best_matches[1][:10].tolist()}...")


def main():
    """Run all tests."""
    print("CLIP with Sequence Similarity - Integration Tests")
    print("=" * 60)

    # Test basic integration
    model = test_clip_integration()

    # Test different similarity types
    similarity_results = test_different_similarity_types()

    # Test sequence analysis
    test_sequence_analysis()

    print("\n" + "=" * 60)
    print("Integration tests completed!")

    # Summary of similarity types
    print(f"\nSimilarity Type Summary:")
    for sim_type, result in similarity_results.items():
        if 'error' not in result:
            print(f"  {sim_type}: total_loss={result['total']:.4f}, seq_loss={result['seq']:.4f}")
        else:
            print(f"  {sim_type}: ERROR - {result['error']}")


if __name__ == "__main__":
    main()