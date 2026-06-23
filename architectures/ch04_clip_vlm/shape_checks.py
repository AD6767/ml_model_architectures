import torch

from architectures.ch04_clip_vlm.contrastive_loss import ClipContrastiveLoss
from architectures.ch04_clip_vlm.mini_clip import MiniCLIP


def check_clip_contrastive_loss():
    batch_size = 4
    embed_dim = 8

    image_embeddings = torch.randn(batch_size, embed_dim)
    text_embeddings = torch.randn(batch_size, embed_dim)

    criterion = ClipContrastiveLoss(temperature=0.07)

    loss, logits_per_image, logits_per_text = criterion(
        image_embed=image_embeddings,
        text_embed=text_embeddings
    )

    print("loss:", loss.item())
    print("logits_per_image:", logits_per_image.shape)
    print("logits_per_text:", logits_per_text.shape)

    assert loss.ndim == 0
    assert logits_per_image.shape == torch.Size([batch_size, batch_size])
    assert logits_per_text.shape == torch.Size([batch_size, batch_size])

def check_clip_contrastive_loss_with_matching_embeddings():
    batch_size = 4
    embed_dim = 8

    embeddings = torch.randn(batch_size, embed_dim)

    criterion = ClipContrastiveLoss(temperature=0.07)

    loss, logits_per_image, logits_per_text = criterion(
        image_embed=embeddings,
        text_embed=embeddings,
    )

    print("matching loss:", loss.item())

    preds = torch.argmax(logits_per_image, dim=1)
    labels = torch.arange(batch_size)

    assert torch.equal(preds.cpu(), labels)
    assert logits_per_image.shape == torch.Size([batch_size, batch_size])
    assert logits_per_text.shape == torch.Size([batch_size, batch_size])

def check_mini_clip_projection_heads():
    batch_size = 4
    image_feature_dim = 128
    text_feature_dim = 256
    embed_dim = 64

    image_features = torch.randn(batch_size, image_feature_dim)
    text_features = torch.randn(batch_size, text_feature_dim)

    model = MiniCLIP(image_feature_dim, text_feature_dim, embed_dim)
    image_embeddings, text_embeddings = model(
        image_features=image_features,
        text_features=text_features,
    )

    print("image_embeddings:", image_embeddings.shape)
    print("text_embeddings:", text_embeddings.shape)
    assert image_embeddings.shape == torch.Size([batch_size, embed_dim])
    assert text_embeddings.shape == torch.Size([batch_size, embed_dim])

    image_norms = image_embeddings.norm(dim=-1)
    text_norms = text_embeddings.norm(dim=-1)
    assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-5)
    assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-5)


def main():
    check_clip_contrastive_loss()
    check_clip_contrastive_loss_with_matching_embeddings()
    check_mini_clip_projection_heads()
    print("CLIP/VLM shape checks passed.")


if __name__ == "__main__":
    main()