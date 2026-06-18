import torch

from patch_embedding import PatchEmbedding


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

def main():
    check_patch_embedding()

if __name__ == '__main__':
    main()
