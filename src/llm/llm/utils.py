"""Utility functions for wave feature processing"""

import torch
from typing import Optional
from torch.nn import CrossEntropyLoss


def compute_causal_lm_loss(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    vocab_size: int,
) -> torch.FloatTensor:
    """
    Compute cross-entropy loss for causal language modeling
    
    Args:
        logits: Model output logits [B, L, V]
        labels: Ground truth labels [B, L]
        vocab_size: Vocabulary size
    
    Returns:
        Loss tensor (scalar)
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def process_wave_features(
    inputs_embeds: torch.FloatTensor,
    input_wave_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    config,
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    """
    Replace wave_patch_tokens with mmwave feature embeddings

    Args:
        inputs_embeds: Text embeddings [B, L, H]
        input_wave_embeds: Mmwave features [B, T, H] (already projected)
        input_ids: Input token ids [B, L]
        attention_mask: Original attention mask [B, L] or None
        config: Model configuration object with wave_patch_token and wave_token_len attributes
    Returns:
        Combined embeddings [B, L_new, H] and updated attention mask [B, L_new]
    """
    batch_size = input_ids.shape[0]
    wave_patch_token_id = getattr(config, 'wave_patch_token', None)
    wave_token_len = getattr(config, 'wave_token_len', 248)

    # Get device from input tensors
    device = inputs_embeds.device

    # Create default attention_mask if None
    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

    # Process each sample in the batch
    new_embeds = []
    new_masks = []

    for batch_idx in range(batch_size):
        sample_embeds = []
        sample_mask = []
        wave_idx = 0

        for seq_idx in range(input_ids.shape[1]):
            current_token_id = input_ids[batch_idx, seq_idx].item()

            if current_token_id == wave_patch_token_id:
                # Replace this wave_patch_token with mmwave embeddings
                if wave_idx < input_wave_embeds.shape[1]:
                    sample_embeds.append(input_wave_embeds[batch_idx, wave_idx])
                    sample_mask.append(1)
                    wave_idx += 1
                else:
                    # No more wave features available, keep original
                    sample_embeds.append(inputs_embeds[batch_idx, seq_idx])
                    sample_mask.append(attention_mask[batch_idx, seq_idx].item())
            else:
                # Regular token, keep original embedding
                sample_embeds.append(inputs_embeds[batch_idx, seq_idx])
                sample_mask.append(attention_mask[batch_idx, seq_idx].item())

        new_embeds.append(torch.stack(sample_embeds))
        new_masks.append(torch.tensor(sample_mask, dtype=torch.long, device=device))

    result_embeds = torch.stack(new_embeds)
    result_masks = torch.stack(new_masks)

    return result_embeds, result_masks

