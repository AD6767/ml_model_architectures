import torch
import torch.nn as nn

from architectures.ch02_transformer_attention.transformer_block import TransformerBlock

from .patch_embedding import PatchEmbedding


class TinyViT(nn.Module):
    """
    Tiny Vision Transformer classifier.

    Input:  [B, 3, 32, 32]
    Output: [B, num_classes]
    """
    def __init__(self, 
                 image_size: int = 32, 
                 patch_size: int = 4, 
                 channels: int = 3, 
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 mlp_ratio: int = 4,
                 num_classes: int = 10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size, 
                                              patch_size=patch_size, 
                                              in_channels=channels, 
                                              embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn([1, 1, embed_dim]) * 0.02) # learnable token [1, 1, embed_dim]
        self.positional_embedding = nn.Parameter(torch.randn([1,
                                                              self.patch_embedding.num_patches + 1, 
                                                              embed_dim]) * 0.02) # learnable positional embedding [1, num_patches + 1(cls), embed_dim]
        self.embed_dim = embed_dim

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(num_blocks)
        ]) # Create TransformerBlock modules.

        self.norm = nn.LayerNorm(embed_dim) # Final normalization before classifier.

        self.mlp_head = nn.Linear(in_features=embed_dim, out_features=num_classes) # Classifier maps CLS embedding to class logits.


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x) # [B, 3, 32, 32] -> [B, 64, embed_dim]
        B, _, embed_dim = x.shape
        cls_token = self.cls_token.expand([B, -1, -1]) # only expand batch dimension, keep the others unchanged.
        x = torch.cat([cls_token, x], dim=1) # [B, 65, embed_dim]
        x = x + self.positional_embedding # [B, 65, embed_dim]

        for block in self.blocks:
            x, _ = block(x) # [B, 65, embed_dim]

        x = self.norm(x) # [B, 65, embed_dim]
        x_cls = x[:,0] # only use cls token for prediction # [B, embed_dim]
        logits = self.mlp_head(x_cls) # [B, num_classes]
        return logits
