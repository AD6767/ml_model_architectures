import math

import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    x: [B, H, N_q, N_k]
    """
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x) # [B, H, N_q, N_k]
    x_exp_sum = torch.sum(x_exp, dim=-1, keepdim=True) # softmax should be over the last dimension
    return x_exp / x_exp_sum # (N, D)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    """
    q: [B, num_heads, seq_len_q, head_dim]
    k: [B, num_heads, seq_len_k, head_dim]
    v: [B, num_heads, seq_len_k, head_dim]

    returns:
        output: attention-weighted values
        attention_weights: softmax scores
    """
    D = q.shape[-1]
    attention_scores = q @ k.transpose(-2, -1) # transpose only last two dimensions of k
                                               # (B, H, N_q, D) @ (B, H, D, N_k) --> (B, H, N_q, N_k)
    scaled_attn_scores = attention_scores / math.sqrt(D) # Without scaling, dot products can become large and make softmax too sharp, which can hurt gradients.
    
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
    
    attention_weights = softmax(scaled_attn_scores) # [B, H, N_q, N_k]
    
    context_vector = attention_weights @ v # [B, H, N_q, N_k] @ [B, H, N_k, D] --> [B, H, N_q, D]
    return context_vector, attention_weights

