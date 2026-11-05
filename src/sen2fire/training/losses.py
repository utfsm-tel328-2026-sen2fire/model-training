from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyFireWeighted(nn.Module):
    """CrossEntropyLoss con pesos por clase para el desbalance fire / non-fire.

    Replica ``Baseline_code/Train.py:134``:

        L_seg = nn.CrossEntropyLoss(weight=torch.Tensor([1, weight]).cuda())

    El vector de pesos se guarda como ``register_buffer`` para que Lightning lo
    mueva al mismo dispositivo del módulo con `.to(device)` automáticamente —
    no hay que recordar `.cuda()` a mano.
    """

    def __init__(self, weight_fire: float = 10.0, n_classes: int = 2) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes debe ser >= 2 (recibido {n_classes})")
        if weight_fire <= 0:
            raise ValueError(f"weight_fire debe ser positivo (recibido {weight_fire})")
        self.n_classes = n_classes
        self.register_buffer(
            "weight",
            torch.tensor([1.0, float(weight_fire)], dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W) ; target: (B,) o (B, H, W) con class indices
        if target.dim() == logits.dim():
            target = target.squeeze(1)
        return F.cross_entropy(logits, target.long(), weight=self.weight)
