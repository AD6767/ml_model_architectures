import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipContrastiveLoss(nn.Module):
    """
    CLIP-style symmetric contrastive loss.

    image_embeddings: [B, D]
    text_embeddings:  [B, D]

    returns:
        loss: scalar
        logits_per_image: [B, B]
        logits_per_text:  [B, B]
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embed: torch.Tensor, text_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize
        image_embed = F.normalize(image_embed, dim=-1) # [B, D]
        text_embed = F.normalize(text_embed, dim=-1) # [B, D]
        # Compute dot product
        logits_per_image = image_embed @ text_embed.T # [B, B]
        # logits_per_text = text_embed @ image_embed.T # [B, B] i.e. transpose of logits_per_image
        logits_per_text = logits_per_image.T
        # Divide by temperature
        logits_per_image = logits_per_image / self.temperature # [B, B]
        logits_per_text = logits_per_text / self.temperature # [B, B]
        B = logits_per_image.shape[0]
        labels = torch.arange(B, device=image_embed.device) # create labels of size B with diagonal 
                                                            # indices as the GT for similarity match.
        image_to_text_loss = F.cross_entropy(logits_per_image, labels)
        text_to_image_loss = F.cross_entropy(logits_per_text, labels)
        loss = (image_to_text_loss + text_to_image_loss) / 2.0
        return loss, logits_per_image, logits_per_text

