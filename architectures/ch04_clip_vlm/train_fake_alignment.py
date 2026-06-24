import torch
import torch.optim as optim

from architectures.ch04_clip_vlm.mini_clip import MiniCLIP
from architectures.ch04_clip_vlm.contrastive_loss import ClipContrastiveLoss

"""
synthetic paired features
-> MiniCLIP projection heads
-> image_embeddings, text_embeddings
-> CLIP contrastive loss
-> optimize projection heads
"""


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_synthetic_batch(
    batch_size: int,
    latent_dim: int,
    image_matrix: torch.Tensor,
    text_matrix: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        image_features: [B, image_feature_dim]
        text_features:  [B, text_feature_dim]
    """
    # 1. Create latent concepts z: [B, latent_dim]
    z = torch.randn(batch_size, latent_dim, device=device)
    # 2. image_features = z @ image_matrix
    image_features = z @ image_matrix # [B, image_feature_dim]
    # 3. text_features = z @ text_matrix
    text_features = z @ text_matrix # [B, text_feature_dim]
    # 4. Add small noise to both
    image_features = image_features + 0.05 * torch.randn_like(image_features)
    text_features = text_features + 0.05 * torch.randn_like(text_features)
    # 5. Return image_features, text_features
    return image_features, text_features


def retrieval_accuracy(logits_per_image: torch.Tensor) -> float:
    """
    logits_per_image: [B (images), B(texts)]
    returns:
        image-to-text retrieval accuracy
    """
    # 1. preds = argmax over text dimension (per row)
    preds = torch.argmax(logits_per_image, dim=1) # (B,)
    # 2. labels = [0, 1, ..., B-1]
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device) # (B,)
    # 3. return mean accuracy
    mean_acc = (preds == labels).float().mean().item()
    return mean_acc


def main():
    device = get_device()
    print(f"Using device: {device}")

    batch_size = 32
    latent_dim = 32
    image_feature_dim = 128
    text_feature_dim = 256
    embed_dim = 64

    model = MiniCLIP(
        image_feature_dim=image_feature_dim,
        text_feature_dim=text_feature_dim,
        embed_dim=embed_dim,
    ).to(device)

    criterion = ClipContrastiveLoss(temperature=0.07)

    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_steps = 200

    image_matrix = torch.randn(latent_dim, image_feature_dim, device=device)
    text_matrix = torch.randn(latent_dim, text_feature_dim, device=device)

    for step in range(num_steps):
        model.train()
        image_features, text_features = create_synthetic_batch(batch_size, 
                                                               latent_dim, 
                                                               image_matrix, 
                                                               text_matrix, 
                                                               device)
        optimizer.zero_grad()
        image_embeddings, text_embeddings = model(image_features, text_features)
        loss, logits_per_image, logits_per_text = criterion(image_embeddings, text_embeddings)
        loss.backward()
        optimizer.step()
        image_to_text_acc = retrieval_accuracy(logits_per_image)
        text_to_image_acc = retrieval_accuracy(logits_per_text)
        if step % 20 == 0 or step == num_steps - 1:
            print(f"step: {step} | loss: {loss.item():.4f} | "
                  f"i2t acc: {image_to_text_acc * 100:.2f} | "
                  f"t2i acc: {text_to_image_acc * 100:.2f}")
    print("Finished fake CLIP alignment training.")


if __name__ == "__main__":
    main()

