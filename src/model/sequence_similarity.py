import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SequenceSimilarity(nn.Module):
    """
    Implements sequence-level similarity calculations for CLIP training.
    Supports multiple similarity strategies:
    1. Global similarity (original CLIP)
    2. Local/window-based similarity
    3. Attention-based similarity
    4. Temporal alignment similarity
    """

    def __init__(
        self,
        embed_dim: int,
        similarity_type: str = "combined",  # "global", "local", "attention", "temporal", "combined"
        window_size: int = 16,
        num_heads: int = 8,
        temperature: float = 0.07,
        weights: dict = None,  # Weights for different similarity components
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.similarity_type = similarity_type
        self.window_size = window_size
        self.temperature = temperature

        # Default weights for different similarity components
        self.weights = weights or {
            'global': 1.0,
            'local': 0.5,
            'attention': 0.3,
            'temporal': 0.2
        }

        # Attention mechanism for attention-based similarity
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Projection layers for different similarity types
        self.global_proj = nn.Linear(embed_dim, embed_dim)
        self.local_proj = nn.Linear(embed_dim, embed_dim)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, radar_seq, text_seq):
        """
        Compute sequence similarity between radar and text sequences.

        Args:
            radar_seq: [b, radar_len, embed_dim]
            text_seq: [b, text_len, embed_dim]

        Returns:
            similarity_scores: [b, b] similarity matrix
        """
        batch_size = radar_seq.size(0)

        if self.similarity_type == "global":
            return self.global_similarity(radar_seq, text_seq)
        elif self.similarity_type == "local":
            return self.local_similarity(radar_seq, text_seq)
        elif self.similarity_type == "attention":
            return self.attention_similarity(radar_seq, text_seq)
        elif self.similarity_type == "temporal":
            return self.temporal_similarity(radar_seq, text_seq)
        elif self.similarity_type == "combined":
            # Combine all similarity types with weights
            similarities = {}

            # Global similarity
            similarities['global'] = self.global_similarity(radar_seq, text_seq)

            # Local similarity
            similarities['local'] = self.local_similarity(radar_seq, text_seq)

            # Attention similarity
            similarities['attention'] = self.attention_similarity(radar_seq, text_seq)

            # Temporal similarity
            similarities['temporal'] = self.temporal_similarity(radar_seq, text_seq)

            # Weighted combination
            combined_sim = sum(self.weights[key] * similarities[key]
                             for key in similarities.keys())

            return combined_sim

        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")

    def global_similarity(self, radar_seq, text_seq):
        """
        Compute global similarity using sequence pooling.

        Args:
            radar_seq: [b, radar_len, embed_dim]
            text_seq: [b, text_len, embed_dim]

        Returns:
            similarity: [b, b]
        """
        # Pool sequences to global representations
        radar_global = radar_seq.mean(dim=1)  # [b, embed_dim]
        text_global = text_seq.mean(dim=1)    # [b, embed_dim]

        # Apply projection
        radar_global = self.global_proj(radar_global)
        text_global = self.global_proj(text_global)

        # Normalize
        radar_global = F.normalize(radar_global, dim=-1)
        text_global = F.normalize(text_global, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(radar_global, text_global.T) / self.temperature

        return similarity

    def local_similarity(self, radar_seq, text_seq):
        """
        Compute local similarity using sliding windows.

        Args:
            radar_seq: [b, radar_len, embed_dim]
            text_seq: [b, text_len, embed_dim]

        Returns:
            similarity: [b, b]
        """
        batch_size = radar_seq.size(0)
        radar_len = radar_seq.size(1)
        text_len = text_seq.size(1)

        # Extract sliding windows
        radar_windows = self.extract_windows(radar_seq, self.window_size)  # [b, num_windows, window_size, embed_dim]
        text_windows = self.extract_windows(text_seq, self.window_size)

        # Pool windows
        radar_pooled = radar_windows.mean(dim=2)  # [b, num_windows, embed_dim]
        text_pooled = text_windows.mean(dim=2)

        # Apply projection
        radar_pooled = self.local_proj(radar_pooled)
        text_pooled = self.local_proj(text_pooled)

        # Normalize
        radar_pooled = F.normalize(radar_pooled, dim=-1)
        text_pooled = F.normalize(text_pooled, dim=-1)

        # Compute pairwise window similarities
        # radar_pooled: [b, n_radar_windows, embed_dim]
        # text_pooled: [b, n_text_windows, embed_dim]

        # Compute similarities between all window pairs for each batch
        # We need to create cross-modal similarity matrix between batch items
        local_similarities = []

        for i in range(batch_size):
            # For each batch item, compute similarity with all other batch items
            batch_similarities = []
            for j in range(batch_size):
                # Compute window similarity between item i and item j
                radar_windows = radar_pooled[i]  # [n_radar_windows, embed_dim]
                text_windows = text_pooled[j]    # [n_text_windows, embed_dim]

                # Compute similarity matrix between windows
                window_sim_matrix = torch.matmul(radar_windows, text_windows.T) / self.temperature  # [n_radar, n_text]

                # Pool across windows to get sequence-level similarity
                similarity_score = window_sim_matrix.max()
                batch_similarities.append(similarity_score)

            local_similarities.append(torch.stack(batch_similarities))

        # Create similarity matrix
        similarity = torch.stack(local_similarities)  # [b, b]

        return similarity

    def attention_similarity(self, radar_seq, text_seq):
        """
        Compute attention-based similarity between sequences.

        Args:
            radar_seq: [b, radar_len, embed_dim]
            text_seq: [b, text_len, embed_dim]

        Returns:
            similarity: [b, b]
        """
        batch_size = radar_seq.size(0)

        # Apply self-attention to each sequence
        radar_attended, _ = self.attention(radar_seq, radar_seq, radar_seq)
        text_attended, _ = self.attention(text_seq, text_seq, text_seq)

        # Apply projection
        radar_attended = self.attention.out_proj(radar_attended)
        text_attended = self.attention.out_proj(text_attended)

        # Compute cross-attention between radar and text
        cross_attention_scores = torch.matmul(radar_attended, text_attended.transpose(-2, -1))  # [b, radar_len, text_len]

        # Use attention weights to compute sequence similarity
        attention_weights = F.softmax(cross_attention_scores / math.sqrt(self.embed_dim), dim=-1)

        # Weighted aggregation of text features based on attention
        attended_text = torch.matmul(attention_weights, text_attended)  # [b, radar_len, embed_dim]

        # Compute similarity between attended sequences
        # Pool to get sequence representations
        radar_pooled = radar_attended.mean(dim=1)  # [b, embed_dim]
        attended_text_pooled = attended_text.mean(dim=1)  # [b, embed_dim]

        # Normalize
        radar_pooled = F.normalize(radar_pooled, dim=-1)
        attended_text_pooled = F.normalize(attended_text_pooled, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(radar_pooled, attended_text_pooled.T) / self.temperature

        return similarity

    def temporal_similarity(self, radar_seq, text_seq):
        """
        Compute temporal similarity by aligning sequences in time.

        Args:
            radar_seq: [b, radar_len, embed_dim]
            text_seq: [b, text_len, embed_dim]

        Returns:
            similarity: [b, b]
        """
        batch_size = radar_seq.size(0)
        radar_len = radar_seq.size(1)
        text_len = text_seq.size(1)

        # Apply projection
        radar_proj = self.temporal_proj(radar_seq)
        text_proj = self.temporal_proj(text_seq)

        # Normalize
        radar_proj = F.normalize(radar_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)

        # Dynamic Time Warping-like alignment
        # Compute similarity matrices for each batch item
        similarities = []

        for i in range(batch_size):
            # Compute pairwise similarities between all time steps
            sim_matrix = torch.matmul(radar_proj[i], text_proj[i].T)  # [radar_len, text_len]

            # Find optimal alignment path using dynamic programming
            # Simplified DTW for efficiency
            alignment_score = self.dtw_alignment(sim_matrix)
            similarities.append(alignment_score)

        # Create similarity matrix
        similarities = torch.stack(similarities)  # [b]
        similarity = similarities.unsqueeze(1).expand(batch_size, batch_size)
        similarity = similarity + similarity.T  # Make symmetric
        similarity = similarity / 2  # Average

        return similarity

    def extract_windows(self, sequence, window_size):
        """
        Extract sliding windows from sequence.

        Args:
            sequence: [b, seq_len, embed_dim]
            window_size: int

        Returns:
            windows: [b, num_windows, window_size, embed_dim]
        """
        batch_size, seq_len, embed_dim = sequence.size()

        if seq_len < window_size:
            # Pad sequence if shorter than window size
            padding = window_size - seq_len
            sequence = F.pad(sequence, (0, 0, 0, padding))
            seq_len = window_size

        num_windows = seq_len - window_size + 1
        windows = []

        for i in range(num_windows):
            windows.append(sequence[:, i:i+window_size, :])

        windows = torch.stack(windows, dim=1)  # [b, num_windows, window_size, embed_dim]

        return windows

    def dtw_alignment(self, sim_matrix):
        """
        Simplified Dynamic Time Warping alignment.

        Args:
            sim_matrix: [seq1_len, seq2_len] similarity matrix

        Returns:
            alignment_score: float
        """
        # Use cumulative max as a simplified DTW for efficiency
        # True DTW would use cumulative min + distance
        cum_sim = torch.cumsum(sim_matrix, dim=0)
        cum_sim = torch.cumsum(cum_sim, dim=1)

        # Return the final alignment score
        return cum_sim[-1, -1] / (sim_matrix.size(0) * sim_matrix.size(1))


class SequenceLoss(nn.Module):
    """
    Loss function that combines sequence similarity with contrastive learning.
    """

    def __init__(
        self,
        embed_dim: int,
        similarity_type: str = "combined",
        temperature: float = 0.07,
        use_siglip: bool = False,
        **kwargs
    ):
        super().__init__()
        self.similarity_fn = SequenceSimilarity(
            embed_dim=embed_dim,
            similarity_type=similarity_type,
            temperature=temperature,
            **kwargs
        )
        self.use_siglip = use_siglip
        self.temperature = temperature

    def forward(self, radar_seq, text_seq, logit_scale=None, logit_bias=None):
        """
        Compute sequence-level contrastive loss.

        Args:
            radar_seq: [b, radar_len, embed_dim]
            text_seq: [b, text_len, embed_dim]
            logit_scale: Optional scale parameter
            logit_bias: Optional bias parameter (for SigLIP)

        Returns:
            loss: Scalar loss
        """
        batch_size = radar_seq.size(0)

        # Compute sequence similarity matrix
        if logit_scale is not None:
            # Apply learnable logit scale if provided
            similarities = self.similarity_fn(radar_seq, text_seq) * logit_scale.exp()
        else:
            similarities = self.similarity_fn(radar_seq, text_seq)

        if logit_bias is not None:
            similarities = similarities + logit_bias

        # Create labels
        labels = torch.arange(batch_size, device=radar_seq.device)

        if self.use_siglip:
            # SigLIP style sigmoid loss
            targets = torch.eye(batch_size, device=radar_seq.device)
            targets = targets * 2 - 1  # Convert to [-1, 1] range

            loss = -F.logsigmoid(targets * similarities).sum() / batch_size
        else:
            # Standard contrastive loss
            loss = F.cross_entropy(similarities, labels)

        return loss