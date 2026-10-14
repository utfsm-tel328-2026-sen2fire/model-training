from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss para segmentación binaria. Espera logits y target {0,1}."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        target = target.float().flatten(1)
        intersection = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    w_dice: float = 0.5,
) -> torch.Tensor:
    """0.5 * Dice + 0.5 * BCEWithLogits. La mezcla estándar para segmentación binaria desbalanceada.

    Acepta target con shape (B, H, W) o (B, 1, H, W).
    """
    if target.dim() == logits.dim() - 1:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    dice = DiceLoss()(logits, target)
    return w_dice * dice + (1.0 - w_dice) * bce
