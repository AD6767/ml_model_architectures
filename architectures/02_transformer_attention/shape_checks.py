import torch

from scaled_dot_product_attention import scaled_dot_product_attention


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

def main():
    check_scaled_dot_product_attention()
    print("Scaled dot-product attention shape checks passed.")

if __name__ == '__main__':
    main()

