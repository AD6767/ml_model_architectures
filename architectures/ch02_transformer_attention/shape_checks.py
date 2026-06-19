import torch

from scaled_dot_product_attention import scaled_dot_product_attention
from multi_head_attention import MultiHeadAttention
from transformer_block import TransformerBlock


def check_scaled_dot_product_attention():
    batch_size = 2
    num_heads = 4
    seq_len_q = 5
    seq_len_k = 6
    head_dim = 8

    q = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len_k, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len_k, head_dim)

    context_vec, attn_weights = scaled_dot_product_attention(q, k, v)

    print("context_vec:", context_vec.shape)
    print("attention_weights:", attn_weights.shape)

    assert context_vec.shape == torch.Size([batch_size, num_heads, seq_len_q, head_dim])
    assert attn_weights.shape == torch.Size([batch_size, num_heads, seq_len_q, seq_len_k])

    attn_weights_row_sums = attn_weights.sum(dim=-1) # should sum to 1.0
    assert torch.allclose(attn_weights_row_sums, torch.ones_like(attn_weights_row_sums), atol=1e-5)

def check_multi_head_attention():
    batch_size = 2
    seq_len = 5
    embed_dim = 32
    num_heads = 4

    x = torch.rand(batch_size, seq_len, embed_dim)

    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    output, attn_weights = mha(x)

    print("mha output:", output.shape)
    print("mha attention_weights:", attn_weights.shape)

    assert output.shape == torch.Size([batch_size, seq_len, embed_dim])
    assert attn_weights.shape == torch.Size([batch_size, num_heads, seq_len, seq_len])

    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    print("Multi-head attention shape checks passed.")

def check_transformer_block():
    batch_size = 2
    seq_len = 5
    embed_dim = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, embed_dim)
    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)

    output, attn_weights = block(x)

    print("transformer block output:", output.shape)
    print("transformer block attention_weights:", attn_weights.shape)

    assert output.shape == torch.Size([batch_size, seq_len, embed_dim])
    assert attn_weights.shape == torch.Size([
        batch_size,
        num_heads,
        seq_len,
        seq_len,
    ])


def main():
    check_scaled_dot_product_attention()
    check_multi_head_attention()
    check_transformer_block()
    print("Transformer attention shape checks passed.")

if __name__ == '__main__':
    main()

