# Transformer Attention

## Architecture
Attention is the core operation used in Transformers. It lets each token look at other tokens and build a contextual representation.
Scaled dot product attention: `softmax(Q @ K.T / sqrt(d_k)) @ V`
Multi-head attention runs attention multiple times in parallel using different learned projections.
```
x
-> Q, K, V projections
-> split into heads
-> scaled attention
-> combine heads
-> Output projection
```

## Problem
RNNs process tokens sequentially and can run into long-range dependencies. Attention allows each token to directly interact with every other token, making it easier to model long-range relationships.

## Input / output
Input to multi-head self attention: x = `[B, seq_len, embed_dim]`
After Q, K, V projections: q, k, v = `[B, seq_len, embed_dim]`
After splitting heads: q, k, v = `[B, num_heads, seq_len, head_dim]` where `head_dim = embed_dim // num_heads`
Attention scores: scores = `[B, num_heads, seq_len_q, seq_len_k]`
Output: context vector (representation) = `[B, seq_len, embed_dim]`

## Key blocks
### Q, K, V projections
The input token embeddings are projected into 3 learned spaces:
Q = what this token is looking for
K = what this token contains
V = what content this token passes forward

For one token:
q = xWq + bq; k = xWk + bk; v = xWv + bv

### Scaled dot-product attention
Compute similarity between queries and keys. Apply softmax over keys. Use weights to mix values. 

### Multi-head attention
Each head attends using a different learned subspace: `[B, seq_len, embed_dim] -> [B, num_heads, seq_len, head_dim]`
After attention, heads are concatenated back: `[B, num_heads, seq_len, head_dim] -> [B, seq_len, embed_dim]`

### Output Projection
After combining back, a linear layer mixes information across heads: `output = combined_heads Wo + bo`. Shape stays the same `[B, seq_len, embed_dim]`

## Loss
Attention itself does not define the loss. The loss depends on the task for eg: cross entropy loss for classification task.

## Why it was introduced
Attention was introduced to model relationships between tokens without processing them strictly sequentially. Transformers use attention to support parallel computation and long-range dependency modeling.

## Strengths
* Models long-range token relationships.
* Parallelizable across sequence positions.
* Works for text, images, audio, video, and multimodal data.
* Foundation for LLMs, ViTs, CLIP, VLMs, and video transformers.

## Weaknesses
* Self-attention cost grows quadratically with sequence length.
* Requires positional information because attention alone is permutation-invariant.
* Can be memory-heavy for long text, high-resolution images, and video.
* Attention weights are not always reliable explanations.
* Needs large data and careful training at scale.

## Production concerns
* Attention can be expensive for long sequences.
* For deployment, consider quantization, KV caching, pruning, and optimized runtimes.
* For video or high-resolution images, reduce token count or use sparse/windowed attention.

## Important questions
1. What does attention compute?
It computes a weighted sum of value vectors, where weights come from query-key similarity.
2. What are Q, K, and V?
Q is what a token is looking for.
K is what each token offers for matching.
V is the content passed forward when a token is attended to.
3. Why do we divide by `sqrt(d_k)`?
Dot products get larger as dimensionality increases. Scaling prevents softmax from becoming too sharp and helps gradients remain stable.
4. Why is softmax applied over the key dimension?
For each query token, we want a probability distribution over all key tokens.
5. What is the shape of attention scores?
For q = `[B, H, Lq, D]`, k = `[B, H, Lk, D]`, the score shape is `[B, H, Lq, Lk]`
6. What is self-attention and cross attention?
Self-attention means Q, K, and V all come from the same input sequence. Cross-attention means Q comes from one sequence, while K and V come from another sequence.
7. Why use multiple heads?
Different heads can learn different relationships, such as syntax, object relations, spatial relations, or long-range dependencies.
8. What does the output projection do?
It mixes the concatenated head outputs and maps them back into the model embedding space.
9. How is causal attention different?
Causal attention masks future tokens so each token can only attend to previous tokens.
10. GELU vs ReLU: why do Transformers usually use GELU?
ReLU fully zeros out negative values. GELU is smoother and gates values based on their magnitude. It does not sharply zero everything below zero. Transformers commonly use GELU because it works well in large deep networks and gives smoother gradients than ReLU.
11. Where is GELU used in a Transformer block?
GELU is usually used inside the MLP / feed-forward block: `Linear -> GeLU -> Linear`. Attention mixes information across tokens. The MLP updates each token representation independently.
12. LayerNorm vs BatchNorm: why do Transformers use LayerNorm?
BatchNorm normalize across batch dimension. LayerNorm normalize across feature dimension. For Transformer input `[B, seq_len, embed_dim]`, LayerNorm normalizes over embed_dim for each token independently. Transformers prefer LayerNorm because sequence lengths can vary, batch sizes may be small, and token-level normalization is more stable for language, vision tokens, and multimodal tokens.
13. Why not use BatchNorm in Transformers?
BatchNorm depends on batch-level statistics, which can be unstable for variable-length sequences, small batches, autoregressive decoding, and distributed training. LayerNorm does not depend on other examples in the batch, so it works better for sequence models.
14. What type of Transformer block did we implement?
We implemented a pre-norm Transformer encoder-style block. It is encoder-style because Q, K, and V all come from the same input sequence. We call it pre-norm because LayerNorm is applied before attention and before the MLP.
15. Encoder block vs decoder block: what is the difference?
encoder block uses bidirectional self-attention. decoder block uses causal self-attention.
16. Why do Transformer blocks use residual connections?
Residual connections help preserve the original representation and improve gradient flow. This makes deep Transformer stacks easier to train. `x = x + attention_output`, `x = x + mlp_output`.
17. What does the MLP do in a Transformer block?
The MLP transforms each token independently after attention. Attention = mixes information across tokens. MLP = mixes information across feature dimensions.
