import torch

from architectures.ch04_clip_vlm.contrastive_loss import ClipContrastiveLoss


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

def main():
    check_clip_contrastive_loss()
    check_clip_contrastive_loss_with_matching_embeddings()
    print("CLIP contrastive loss shape checks passed.")


if __name__ == "__main__":
    main()