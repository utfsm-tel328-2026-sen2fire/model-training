"""Tests del UNet configurable."""
from __future__ import annotations

import pytest
import torch

from sen2fire.models import UNet


def test_unet_depth_5_bilinear_true_forward_shape() -> None:
    """Defaults = replican al paper y devuelven (B, 2, H, W)."""
    m = UNet()
    y = m(torch.randn(1, 12, 64, 64))
    assert y.shape == (1, 2, 64, 64)


def test_unet_depth_5_bilinear_false_forward_shape() -> None:
    m = UNet(bilinear=False)
    y = m(torch.randn(1, 12, 64, 64))
    assert y.shape == (1, 2, 64, 64)


def test_unet_in_channels_13() -> None:
    """Modo 1 de los autores: 12 bandas + aerosol = 13."""
    m = UNet(in_channels=13)
    y = m(torch.randn(2, 13, 32, 32))
    assert y.shape == (2, 2, 32, 32)


def test_unet_bilinear_ch_factor_halves() -> None:
    """En modo bilinear los canales al final del U son menores (factor=2)."""
    bilinear = UNet()
    convt = UNet(bilinear=False)
    p_bi = sum(p.numel() for p in bilinear.parameters())
    p_co = sum(p.numel() for p in convt.parameters())
    # ConvT tiene muchos más parámetros que Bilinear por los kernels 2x2.
    assert p_co > p_bi


def test_unet_n_classes_must_be_at_least_2() -> None:
    """Replicamos el setup binario (fire / non-fire)."""
    with pytest.raises(ValueError, match="n_classes"):
        UNet(n_classes=1)


def test_unet_depth_paper_only() -> None:
    """Sólo se soporta depth=5 hoy."""
    with pytest.raises(ValueError, match="depth debe ser 5"):
        UNet(depth=4)
    with pytest.raises(ValueError, match="depth debe ser 5"):
        UNet(depth=6)


def test_unet_gradient_flows() -> None:
    m = UNet()
    x = torch.randn(1, 12, 32, 32, requires_grad=True)
    y = m(x)
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(y, target)
    loss.backward()
    # Algún parámetro del inc debe tener grad no nulo.
    inc_grad = m.inc.net[0].weight.grad
    assert inc_grad is not None and torch.any(inc_grad != 0).item()
