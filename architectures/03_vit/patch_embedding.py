import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Convert image into patch tokens.

    Input:  [B, C, H, W]
    Output: [B, num_patches, embed_dim]
    """
    def __init__(self, 
                 image_size: int = 32,
                 patch_size: int = 4,
                 in_channels: int = 3,
                 embed_dim: int = 64):
        super().__init__()
        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.embed_dim = embed_dim

        # [B, 3, 32, 32] -> [B, embed_dim, 8, 8]
        # output size = ((I - K + 2*P) / S) + 1
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(4,4), stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]

        returns:
            patch_tokens: [B, num_patches, embed_dim]
        """
        x = self.conv(x) # [B, C, H, W] -> [B, embed_dim, 8, 8]
        B = x.shape[0]
        x = x.reshape([B, self.embed_dim, -1]) # or x.flatten(start_dim=2) [B, embed_dim, 8, 8] -> [B, embed_dim, 64]
        x = x.transpose(1, 2) # [B, embed_dim, 64] -> [B, 64, embed_dim]
        return x
