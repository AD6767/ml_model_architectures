import torch

from patch_embedding import PatchEmbedding
from tiny_vit import TinyViT


def check_patch_embedding():
    patch_size = 4
    batch_size = 2
    image_size = 32
    embed_dim = 64
    channels = 3
    x = torch.randn([batch_size, channels, image_size, image_size])
    patch_embed = PatchEmbedding(image_size=image_size, 
                                 patch_size=patch_size,
                                 in_channels=channels,
                                 embed_dim=embed_dim)
    out = patch_embed(x)
    
    print("patch tokens:", out.shape)
    expected_num_patches = (image_size // patch_size) ** 2
    assert out.shape == torch.Size([batch_size, expected_num_patches, embed_dim])
    print("Patch embedding shape checks passed.")

def check_tiny_vit_token_preparation():
    batch_size = 2
    image_size = 32
    patch_size = 4
    embed_dim = 64
    x = torch.randn([batch_size, 3, image_size, image_size])
    model = TinyViT(image_size=image_size,
                    patch_size=patch_size,
                    channels=3,
                    embed_dim=embed_dim)
    tokens = model(x)
    print("tokens with cls + pos:", tokens.shape)
    expected_num_patches = (image_size // patch_size) ** 2
    assert tokens.shape == torch.Size([
        batch_size,
        expected_num_patches + 1,
        embed_dim,
    ])
    print("TinyViT token preparation shape checks passed.")


def main():
    check_patch_embedding()
    check_tiny_vit_token_preparation()

if __name__ == '__main__':
    main()
