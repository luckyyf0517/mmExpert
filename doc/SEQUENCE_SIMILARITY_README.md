# CLIP Sequence Similarity Implementation

This document describes the sequence similarity enhancement to the CLIP model that addresses the sequence information loss problem identified in the analysis.

## Overview

The enhanced CLIP implementation supports sequence-level similarity computation, allowing the model to preserve and leverage temporal information in radar data and sequential structure in text data. This addresses the key limitation where the original implementation only used EOT (End of Token) representations, discarding valuable sequence information.

## Key Features

### 1. Sequence-Level Encoding
- **Radar encoder**: Returns full feature sequences instead of just EOT tokens
- **Text encoder**: Returns full token sequences instead of pooled representations
- **Backward compatibility**: Maintains support for original pooled representations

### 2. Multiple Similarity Strategies

#### Global Similarity
- Uses sequence pooling (mean, max, etc.) to create global representations
- Computationally efficient
- Good for capturing overall content similarity

#### Local Similarity
- Uses sliding windows to capture local patterns
- Window size configurable (default: 16)
- Preserves spatial and temporal locality

#### Attention-Based Similarity
- Uses multi-head attention for cross-modal alignment
- Learns which parts of sequences should align
- Most flexible but computationally expensive

#### Temporal Similarity
- Uses Dynamic Time Warping-like alignment
- Aligns sequences in time dimension
- Good for temporal data with different speeds

#### Combined Similarity (Recommended)
- Combines all similarity types with learnable weights
- Default weights: global=1.0, local=0.5, attention=0.3, temporal=0.2
- Most comprehensive approach

### 3. Flexible Loss Computation
- Standard CLIP loss (global representations)
- Sequence similarity loss (sequence representations)
- Weighted combination of both losses
- Supports both standard CLIP and SigLIP loss functions

## Usage

### Basic Configuration

```python
# Create CLIP model with sequence similarity
model = CLIP(
    encoder_cfg=encoder_cfg,
    text_cfg=text_cfg,
    context_length=512,
    transformer_width=512,
    transformer_layers=6,
    transformer_heads=8,
    temperature=0.07,
    # Sequence similarity settings
    use_sequence_similarity=True,
    sequence_similarity_type="combined",
    sequence_similarity_weight=0.5
)
```

### Training

```python
# Standard training loop - sequence similarity is handled automatically
for batch in dataloader:
    loss = model.training_step(batch, batch_idx)
    loss.backward()
    optimizer.step()
```

### Sequence Feature Extraction

```python
# Get sequence features for analysis
radar_seq, text_seq = model.encode_sequences(radar_data, text)

# Get both pooled and sequence features
radar_pooled, text_pooled, radar_seq, text_seq = model.forward(
    radar_data, text, return_sequences=True
)
```

## Configuration Options

### Similarity Types
- `"global"`: Only global similarity (most efficient)
- `"local"`: Only local/window-based similarity
- `"attention"`: Only attention-based similarity
- `"temporal"`: Only temporal alignment similarity
- `"combined"`: Weighted combination of all types (recommended)

### Sequence Loss Weight
- `sequence_similarity_weight`: Controls contribution of sequence loss
- Typical range: 0.1 to 1.0
- Default: 0.5 (equal weight with standard CLIP loss)

### Window Size (for local similarity)
- `window_size`: Size of sliding windows
- Default: 16
- Larger windows capture more context but less locality

## Performance Considerations

### Memory Usage
- Sequence features require more memory than pooled features
- Use smaller batch sizes or gradient accumulation if needed
- Consider using `"global"` similarity type for memory-constrained environments

### Computation Time
- Attention-based similarity is most computationally expensive
- Local similarity adds moderate overhead
- Combined similarity includes all computations

### Training Stability
- Start with lower `sequence_similarity_weight` (0.1-0.3) and gradually increase
- Monitor both `loss_clip` and `loss_seq` separately
- Use gradient clipping if training becomes unstable

## Monitoring and Debugging

### Loss Components
The model logs multiple loss components:
- `loss_clip`: Standard CLIP loss
- `loss_seq`: Sequence similarity loss
- `loss_total`: Combined loss

### Feature Statistics
Enable feature norm logging to monitor training:
```python
model._log_norm_stats = True
```

### Sequence Analysis
After training, analyze sequence features:
```python
# Temporal variation analysis
radar_std = torch.std(radar_seq, dim=1)
text_std = torch.std(text_seq, dim=1)

# Similarity analysis
similarity_fn = model.sequence_loss_fn.similarity_fn
similarity_matrix = similarity_fn(radar_seq, text_seq)
```

## Integration with Existing Code

### Backward Compatibility
- All existing code continues to work unchanged
- Sequence features are only computed when `use_sequence_similarity=True`
- Original EOT-based encoding remains the default

### Migration Path
1. Start with `use_sequence_similarity=False` to verify baseline
2. Enable with `sequence_similarity_type="global"` for minimal overhead
3. Progress to `"combined"` for full benefits
4. Tune `sequence_similarity_weight` based on validation performance

## Expected Benefits

Based on the analysis document, this implementation should provide:

1. **Better Information Retention**: Preserves temporal dynamics and local features
2. **Improved Alignment**: Enables fine-grained cross-modal alignment
3. **Enhanced Performance**: Better performance on temporal tasks
4. **Richer Representations**: More comprehensive feature representations

## Example Results

When training with sequence similarity enabled, you should observe:
- Lower sequence loss over time
- Better temporal alignment in visualizations
- Improved performance on time-sensitive tasks
- More interpretable attention patterns (if using attention similarity)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use `"global"` similarity
2. **Training Instability**: Reduce `sequence_similarity_weight` or use gradient clipping
3. **Slow Training**: Consider using fewer similarity types or smaller windows
4. **Poor Convergence**: Start with pretrained weights and gradually introduce sequence loss

### Performance Tips

1. Use mixed precision training (`torch.cuda.amp`)
2. Enable gradient checkpointing for large models
3. Use appropriate window sizes based on your data characteristics
4. Monitor GPU memory usage and adjust batch size accordingly