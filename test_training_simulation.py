#!/usr/bin/env python3
"""
Simulate a complete training scenario with sequence similarity CLIP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.sequence_similarity import SequenceSimilarity, SequenceLoss


class MockRadarEncoder(nn.Module):
    """Mock radar encoder."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.conv = nn.Conv1d(256, embed_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, radar_data):
        x = radar_data['range_time']  # [b, 256, 496]
        x = self.conv(x)  # [b, embed_dim, 496]
        x = x.transpose(1, 2)  # [b, 496, embed_dim]
        x, _ = self.lstm(x)  # [b, 496, embed_dim]
        x = self.proj(x)

        pooled = x.mean(dim=1)  # [b, embed_dim]
        pooled = F.normalize(pooled, dim=-1)

        return pooled, x


class MockTextEncoder(nn.Module):
    """Mock text encoder."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(1000, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text):
        batch_size = len(text)
        seq_len = 32  # Fixed sequence length for mock encoder

        # Mock tokenization
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))
        x = self.embedding(token_ids)
        x = self.transformer(x)
        x = self.proj(x)

        pooled = x[:, -1, :]  # [b, embed_dim]
        pooled = F.normalize(pooled, dim=-1)

        return pooled, x


class MockSequenceCLIP(nn.Module):
    """Mock CLIP model with sequence similarity."""
    def __init__(self, embed_dim=512, sequence_similarity_weight=0.5):
        super().__init__()
        self.radar_encoder = MockRadarEncoder(embed_dim)
        self.text_encoder = MockTextEncoder(embed_dim)
        self.sequence_similarity_weight = sequence_similarity_weight

        # Sequence loss function
        self.sequence_loss_fn = SequenceLoss(
            embed_dim=embed_dim,
            similarity_type="combined",
            temperature=0.07
        )

        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def forward(self, radar_data, text, return_sequences=False):
        radar_pooled, radar_seq = self.radar_encoder(radar_data)
        text_pooled, text_seq = self.text_encoder(text)

        if return_sequences:
            return radar_pooled, text_pooled, radar_seq, text_seq
        return radar_pooled, text_pooled

    def training_step(self, batch, batch_idx):
        """Training step with sequence similarity."""
        # Prepare data
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
        logits = torch.matmul(radar_pooled, text_pooled.T) * self.logit_scale.exp()
        labels = torch.arange(radar_pooled.size(0), device=radar_pooled.device)
        loss_clip = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)

        # Sequence similarity loss
        loss_seq = self.sequence_loss_fn(radar_seq, text_seq)

        # Combined loss
        total_loss = (1 - self.sequence_similarity_weight) * loss_clip + \
                    self.sequence_similarity_weight * loss_seq

        return {
            'loss_total': total_loss,
            'loss_clip': loss_clip,
            'loss_seq': loss_seq,
            'logits': logits
        }

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        with torch.no_grad():
            return self.training_step(batch, batch_idx)


def create_synthetic_dataset(num_samples=100, batch_size=8):
    """Create synthetic dataset for testing."""
    batches = []

    for i in range(0, num_samples, batch_size):
        actual_batch_size = min(batch_size, num_samples - i)

        batch = {
            'input_wave_range': torch.randn(actual_batch_size, 256, 496),
            'input_wave_doppler': torch.randn(actual_batch_size, 128, 496),
            'input_wave_azimuth': torch.randn(actual_batch_size, 128, 496),
            'caption': [
                f"Sample description {j}" for j in range(actual_batch_size)
            ]
        }
        batches.append(batch)

    return batches


def simulate_training():
    """Simulate a training session."""
    print("Simulating Training with Sequence Similarity CLIP")
    print("=" * 60)

    # Create model
    model = MockSequenceCLIP(embed_dim=512, sequence_similarity_weight=0.5)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create synthetic dataset
    train_batches = create_synthetic_dataset(num_samples=64, batch_size=8)
    val_batches = create_synthetic_dataset(num_samples=16, batch_size=4)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_batches) * 8}")
    print(f"Validation samples: {len(val_batches) * 4}")
    print(f"Sequence similarity weight: {model.sequence_similarity_weight}")

    # Training loop
    num_epochs = 5
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Training
        model.train()
        epoch_train_losses = []

        for batch_idx, batch in enumerate(train_batches):
            optimizer.zero_grad()

            # Forward pass
            outputs = model.training_step(batch, batch_idx)

            # Backward pass
            outputs['loss_total'].backward()
            optimizer.step()

            epoch_train_losses.append(outputs['loss_total'].item())

            if batch_idx % 4 == 0:
                print(f"  Batch {batch_idx}: total={outputs['loss_total'].item():.4f}, "
                      f"clip={outputs['loss_clip'].item():.4f}, "
                      f"seq={outputs['loss_seq'].item():.4f}")

        # Validation
        model.eval()
        epoch_val_losses = []

        with torch.no_grad():
            for batch in val_batches:
                outputs = model.validation_step(batch, 0)
                epoch_val_losses.append(outputs['loss_total'].item())

        # Store losses
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {avg_val_loss:.4f}")

    return model, train_losses, val_losses


def analyze_learned_representations(model, num_samples=4):
    """Analyze the learned representations."""
    print("\n\nAnalyzing Learned Representations")
    print("=" * 60)

    model.eval()

    # Create test data
    test_batch = {
        'input_wave_range': torch.randn(num_samples, 256, 496),
        'input_wave_doppler': torch.randn(num_samples, 128, 496),
        'input_wave_azimuth': torch.randn(num_samples, 128, 496),
        'caption': [f"Test sample {i}" for i in range(num_samples)]
    }

    radar_data = {
        'range_time': test_batch['input_wave_range'],
        'doppler_time': test_batch['input_wave_doppler'],
        'azimuth_time': test_batch['input_wave_azimuth']
    }

    with torch.no_grad():
        # Get features
        radar_pooled, text_pooled, radar_seq, text_seq = model.forward(
            radar_data, test_batch['caption'], return_sequences=True
        )

        # Analyze similarity matrix
        logits = torch.matmul(radar_pooled, text_pooled.T) * model.logit_scale.exp()

        print(f"Radar pooled shape: {radar_pooled.shape}")
        print(f"Text pooled shape: {text_pooled.shape}")
        print(f"Radar sequence shape: {radar_seq.shape}")
        print(f"Text sequence shape: {text_seq.shape}")

        print(f"\nSimilarity matrix (logits):")
        print(logits.numpy())

        print(f"\nDiagonal (correct matches): {torch.diag(logits).tolist()}")
        print(f"Off-diagonal mean: {((logits.sum() - torch.diag(logits).sum()) / (num_samples**2 - num_samples)):.4f}")

        # Analyze sequence similarity
        seq_sim_fn = model.sequence_loss_fn.similarity_fn
        seq_similarities = seq_sim_fn(radar_seq, text_seq)

        print(f"\nSequence similarity matrix:")
        print(seq_similarities.numpy())
        print(f"Sequence diagonal: {torch.diag(seq_similarities).tolist()}")

        # Analyze feature statistics
        print(f"\nFeature Statistics:")
        print(f"Radar norm mean: {torch.norm(radar_pooled, dim=1).mean().item():.4f}")
        print(f"Text norm mean: {torch.norm(text_pooled, dim=1).mean().item():.4f}")

        # Temporal dynamics
        radar_temporal_var = torch.std(radar_seq, dim=1).mean()
        text_temporal_var = torch.std(text_seq, dim=1).mean()

        print(f"Radar temporal variation: {radar_temporal_var.item():.4f}")
        print(f"Text temporal variation: {text_temporal_var.item():.4f}")

    return radar_pooled, text_pooled, radar_seq, text_seq


def test_ablation_study():
    """Test different sequence similarity weights."""
    print("\n\nAblation Study: Sequence Similarity Weights")
    print("=" * 60)

    weights = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}

    test_batch = create_synthetic_dataset(num_samples=16, batch_size=4)[0]

    for weight in weights:
        print(f"\nTesting weight = {weight:.1f}")

        # Create model with specific weight
        model = MockSequenceCLIP(embed_dim=256, sequence_similarity_weight=weight)
        model.eval()

        with torch.no_grad():
            outputs = model.training_step(test_batch, 0)

            total_loss = outputs['loss_total'].item()
            clip_loss = outputs['loss_clip'].item()
            seq_loss = outputs['loss_seq'].item()

            print(f"  Total loss: {total_loss:.4f}")
            print(f"  CLIP loss:  {clip_loss:.4f}")
            print(f"  Seq loss:   {seq_loss:.4f}")

            # Check if weight is correctly applied
            expected = (1 - weight) * clip_loss + weight * seq_loss
            print(f"  Expected:   {expected:.4f}")
            print(f"  Difference: {abs(total_loss - expected):.6f}")

            results[weight] = {
                'total': total_loss,
                'clip': clip_loss,
                'seq': seq_loss
            }

    return results


def main():
    """Run all tests."""
    print("Sequence Similarity CLIP - Complete Training Simulation")
    print("=" * 70)

    # Training simulation
    model, train_losses, val_losses = simulate_training()

    # Analyze learned representations
    analyze_learned_representations(model)

    # Ablation study
    ablation_results = test_ablation_study()

    print("\n" + "=" * 70)
    print("Simulation completed!")

    # Summary
    print(f"\nFinal training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Training improvement: {train_losses[0] - train_losses[-1]:.4f}")

    # Best weight from ablation
    best_weight = min(ablation_results.keys(), key=lambda w: ablation_results[w]['total'])
    print(f"\nBest sequence similarity weight: {best_weight}")
    print(f"Best total loss: {ablation_results[best_weight]['total']:.4f}")


if __name__ == "__main__":
    main()