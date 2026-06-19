# Vision Transformer

## Architecture
Vision Transformer treats image as a sequence of patch tokens. `image -> patch_embedding -> cls_token -> positional_embedding`.
For a `32x32` rgb image with `patch_size=4`:
num_patches = (32/4) * (32/4) = 64. patch_dim = 3 * 4 * 4 = 48.
Each raw patch is projected into embedding vector: `B, 3, 32, 32` -> `B, 64, embed_dim`

## Problem
CNNs use convolutional inductive biases such as locality and translation equivariance. ViT asks whether images can be modeled as token sequences using Transformers, similar to NLP. The key idea is that image patches behave like tokens.

## Input / output
Input: `B, 3, 32, 32`
Output: `[B, num_classes]`
Patch Embedding: `[B, 3, 32, 32]` -> `[B, 64, embed_dim]`
After adding CLS token: `[B, 64, embed_dim]` -> `[B, 65, embed_dim]`
After adding positional embedding: `[B, 65, embed_dim]` -> `[B, 65, embed_dim]`
Classifier (MLP): cls_token: `[B, embed_dim]`, logits: `[B, num_classes]`

## Key blocks
### Patch Embedding
Patch embedding converts image patches into token embeddings.
Conceptually: `[B, 3, 32, 32]` -> `[B, 64, 48]` -> `Linear(48, embed_dim)` -> `[B, 64, embed_dim]`
Practical Implementation: Using `Conv2d`. This performs patch extraction and linear projection together.

### CLS token
CLS token is a learned token prepended to the patch sequence `[B, 64, embed_dim]` -> `[B, 65, embed_dim]`
At the end of the Transformer, the CLS token is used for classification. `x_cls` = `x[:, 0]`

### Positional embedding
Self-attention does not know patch order or spatial location by itself. Positional embeddings are added to patch tokens so the model can learn where each patch came from. `tokens = tokens + positional_embedding`. Shape = `[1, num_patches + 1, embed_dim]`

### Transformer blocks
ViT uses Transformer encoder-style blocks.
```
LayerNorm -> Self-Attention -> Residual
LayerNorm -> MLP -> Residual
```
Self-attention mixes information across image patches. MLP updates each patch representation independently.

### Classification head
After Transformer blocks: `x: [B, 65, embed_dim]`
Use CLS token: `x_cls: [B, embed_dim]`
Then linear classifier: `logits: [B, num_classes]`

## Loss
For image classification, ViT commonly uses cross-entropy loss. `logits: [B, num_classes], labels: [B]`

## Why it was introduced
ViT showed that pure Transformer architectures can work well for computer vision when images are represented as patch tokens. It reduced reliance on convolution-specific design and made it easier to reuse Transformer scaling ideas from NLP.

## Strengths
* Models global relationships between image patches.
* Scales well with large datasets and pretraining.
* Useful foundation for CLIP, VLMs, video transformers, and multimodal models.
* Flexible token-based design.

## Weaknesses
* Less data-efficient than CNNs when trained from scratch.
* Self-attention cost grows quadratically with number of patches.
* Can be expensive for high-resolution images.
* Patch embedding may lose fine-grained local detail.
* Requires positional embeddings to understand spatial layout.

## Production concerns
* Token count strongly affects latency and memory. Smaller patch size gives more tokens and higher cost. Larger patch size is faster but may lose detail.
* Use smaller embed_dim, fewer heads, or fewer blocks for edge deployment.

## Important questions
1. What is the main idea of ViT?
ViT converts an image into a sequence of patch tokens and processes them using Transformer blocks.
2. Why does patch embedding use Conv2d?
A convolution with kernel_size = patch_size and stride = patch_size extracts non-overlapping patches and projects each patch into embed_dim. It is equivalent to patch extraction plus a linear layer.
3. What is the CLS token?
The CLS token is a learned token added to the beginning of the patch sequence. It collects global image information through self-attention and is used for classification.
4. Why do we need positional embeddings?
Self-attention is permutation-invariant. Without positional embeddings, the model does not know where each patch came from in the image.
5. What type of Transformer block does ViT use?
Standard ViT uses Transformer encoder-style blocks with bidirectional self-attention. Each patch token can attend to every other patch token.
6. How is ViT different from ResNet?
ResNet processes images as spatial feature maps using convolutions. ViT processes images as sequences of patch tokens using self-attention. ResNet: [B, C, H, W], ViT: [B, num_patches, embed_dim]
7. Why can ViT be expensive?
Self-attention compares every token with every other token. For N patches, attention cost ~O(N^2).
8. Why does ViT need more data than CNNs?
CNNs have strong image-specific inductive biases such as locality and translation equivariance. ViT has weaker image-specific bias, so it often benefits more from large-scale pretraining.