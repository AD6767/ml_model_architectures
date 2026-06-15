import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention

"""
x
-> LayerNorm
-> MultiHeadAttention
-> residual add

-> LayerNorm
-> MLP / FeedForward
-> residual add
"""

class TransformerBlock(nn.Module):
    """
    Tiny pre-norm Transformer block.

    Input:  [B, seq_len, embed_dim]
    Output: [B, seq_len, embed_dim]
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, seq_len, embed_dim]

        returns:
            x: [B, seq_len, embed_dim]
            attention_weights: [B, num_heads, seq_len, seq_len]
        """
        attn_output, attn_weights = self.mha(self.norm1(x), mask=mask)
        x = x + attn_output
        
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, attn_weights



