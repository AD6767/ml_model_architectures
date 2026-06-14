import torch
import torch.nn as nn

from scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.

    Input:  [B, seq_len, embed_dim]
    Output: [B, seq_len, embed_dim]
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Define linear layers for q, k, v projections (create weight matrices)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, seq_len, embed_dim]
        Output: [B, num_heads, seq_len, head_dim]
        """
        # 1. Get B, L, E from x.shape
        B, L, E = x.shape
        # 2. Reshape to [B, L, num_heads, head_dim]
        x = x.reshape(B, L, self.num_heads, self.head_dim)
        # 3. Transpose to [B, num_heads, L, head_dim]
        x = x.transpose(1, 2)
        return x
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, num_heads, seq_len, head_dim]
        Output: [B, seq_len, embed_dim]
        """
        B, H, L, D = x.shape
        # 1. Transpose to [B, seq_len, num_heads, head_dim]
        x = x.transpose(1, 2)
        # 2. Make tensor contiguous if needed (If you use .view(), you must manually use .contiguous() after a transpose)
        # 3. Reshape to [B, seq_len, embed_dim]
        x = x.reshape(B, L, H * D)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, seq_len, embed_dim]

        returns:
            output: [B, seq_len, embed_dim]
            attention_weights: [B, num_heads, seq_len, seq_len]
        """
        # 1. Project x into q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # 2. Split heads
        q = self._split_heads(q) # B, H, L, D
        k = self._split_heads(k) # B, H, L, D
        v = self._split_heads(v) # B, H, L, D
        # 3. Call scaled_dot_product_attention
        context_vec_matrices, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        # 4. Combine heads
        combined = self._combine_heads(context_vec_matrices) # B, L, E
        # 5. Apply output projection
        out = self.out_proj(combined) # B, L, E
        # 6. Return output and attention weights
        return out, attn_weights

