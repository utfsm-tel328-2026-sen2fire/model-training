"""Fixtures compartidas de los tests.

Todo corre contra directorios temporales y datos sintéticos: sin red, sin GPU
y sin tocar el `.env` real.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from make_fake_data import make_fake_scene


def make_scene(
    root: Path, *, scenes: int = 4, per_scene: int = 4, size: int = 32
) -> Path:
    """Crea un árbol `scene1/..sceneN/` con patches .npz sintéticos."""
    for s in range(1, scenes + 1):
        make_fake_scene(root / f"scene{s}", per_scene, size, seed=s)
    return root


@pytest.fixture
def fake_src(tmp_path: Path) -> Path:
    """Dataset Sen2Fire sintético: 4 escenas x 4 patches de 32x32."""
    return make_scene(tmp_path / "src")