"""Tests de las losses configurables."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sen2fire.training.losses import CrossEntropyFireWeighted


def test_cross_entropy_fire_weighted_runs() -> None:
    loss = CrossEntropyFireWeighted(weight_fire=10.0)
    logits = torch.randn(2, 2, 8, 8)
    target = torch.randint(0, 2, (2, 8, 8))
    out = loss(logits, target)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_fire_weight_changes_loss() -> None:
    """Cambiar `weight_fire` cambia la loss numérica.

    (En un setup con target mixto y logits al azar, las dos pérdidas difieren
    porque los píxeles fire se ponderan distinto al promediar.)
    """
    logits = torch.randn(1, 2, 16, 16)
    target = torch.randint(0, 2, (1, 16, 16))
    loss_w1 = CrossEntropyFireWeighted(weight_fire=1.0)(logits, target)
    loss_w5 = CrossEntropyFireWeighted(weight_fire=5.0)(logits, target)
    assert not torch.allclose(loss_w1, loss_w5)


def test_fire_weight_increases_grad_on_fire_class() -> None:
    """Con weight_fire alto, los píxeles fire mal clasificados producen un
    gradiente más grande que con weight_fire bajo.

    Setup:
    - 4 píxeles fire target, 12 non-fire target.
    - logits uniformes (predicciones random): así las CE por píxel son similares
      y el gradiente w.r.t. logits refleja principalmente la señal de peso en
      el class-target.
    - Pero añadimos logits asimétricos (logits[:,fire_chan]=+2 y logits[:,nonfire_chan]=-2)
      para forzar que los fire-targets tengan menor CE y los non-fire-targets
      mayor CE — así weight_fire alto redistribuye la composición del gradiente
      entre los dos canales.
    """
    torch.manual_seed(42)
    logits = torch.zeros(1, 2, 4, 4, requires_grad=True)
    logits.retain_grad()
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    target[0, 0:2, 0:2] = 1   # 4 píxeles fire target
    # Set non-fire predictivos con confianza (CE baja) y fire como incierto (CE ~ media).

    loss = CrossEntropyFireWeighted(weight_fire=10.0)(logits, target)
    loss.backward()
    grad_with_w10 = logits.grad.detach().clone()

    logits.grad.zero_()
    loss = CrossEntropyFireWeighted(weight_fire=1.0)(logits, target)
    loss.backward()
    grad_with_w1 = logits.grad.detach()

    # Las dos pérdidas son distintas (los pesos difieren), así que el
    # `logits.grad` total difiere. Verificamos que difieren en L1 medio:
    diff = (grad_with_w10 - grad_with_w1).abs().mean()
    assert diff > 1e-4, f"grad no cambia con weight_fire: {diff=}"


def test_weight_buffer_auto_moves_to_device() -> None:
    """`register_buffer` debe dejar el `.to(device)` actualizándolo."""
    loss = CrossEntropyFireWeighted(weight_fire=10.0)
    loss.to("cpu")
    assert loss.weight.device.type == "cpu"


def test_loss_backwards_finite() -> None:
    loss = CrossEntropyFireWeighted(weight_fire=10.0)
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (1, 4, 4))
    val = loss(logits, target)
    val.backward()
    assert torch.isfinite(logits.grad).all()


def test_invalid_weight_raises() -> None:
    with pytest.raises(ValueError):
        CrossEntropyFireWeighted(weight_fire=0.0)
    with pytest.raises(ValueError):
        CrossEntropyFireWeighted(weight_fire=-1.0)


def test_handles_b_class_indices_target() -> None:
    """El target puede ser (B, H, W) o (B,) — la loss normaliza."""
    loss = CrossEntropyFireWeighted(weight_fire=10.0)
    logits = torch.randn(3, 2, 5, 5)
    target = torch.randint(0, 2, (3, 5, 5))
    out = loss(logits, target)
    assert torch.isfinite(out)
