import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Project encoder features into shared CLIP embedding space.

    Input:  [B, input_dim]
    Output: [B, embed_dim]
    """
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_features=input_dim, out_features=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # [B, input_dim] -> [B, embed_dim]
        x = F.normalize(x, dim=-1) # Normalize output along dim=-1
        return x


class MiniCLIP(nn.Module):
    """
    Tiny CLIP-style model using precomputed image/text features.

    image_features: [B, image_feature_dim]
    text_features:  [B, text_feature_dim]

    returns:
        image_embeddings: [B, embed_dim]
        text_embeddings:  [B, embed_dim]
    """
    def __init__(self, image_feature_dim: int, text_feature_dim: int, embed_dim: int):
        super().__init__()
        self.image_projection = ProjectionHead(image_feature_dim, embed_dim)
        self.text_projection = ProjectionHead(text_feature_dim, embed_dim)

    def forward(self, 
                image_features: torch.Tensor, 
                text_features: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embedding = self.image_projection(image_features) # [B, image_feature_dim] -> [B, embed_dim]
        text_embedding = self.text_projection(text_features) # [B, text_feature_dim] -> [B, embed_dim]
        return image_embedding, text_embedding
