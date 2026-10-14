from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import resolve_root
from .config import load_config


def _scene_to_idx(samples: list[Path]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, p in enumerate(samples):
        scene = p.parent.name
        out.setdefault(scene, []).append(i)
    return out


def get_scene_split(
    samples: list[Path],
    train: tuple[str, ...] = ("scene1", "scene2"),
    val: tuple[str, ...] = ("scene3",),
    test: tuple[str, ...] = ("scene4",),
) -> dict[str, list[int]]:
    by_scene = _scene_to_idx(samples)

    def _gather(names: tuple[str, ...]) -> list[int]:
        idx: list[int] = []
        for n in names:
            if n not in by_scene:
                raise KeyError(f"escena {n!r} no encontrada; disponibles: {sorted(by_scene)}")
            idx.extend(by_scene[n])
        return idx

    return {"train": _gather(train), "val": _gather(val), "test": _gather(test)}


class Sen2FireDataset(Dataset):
    """Dataset Sen2Fire.

    Cada muestra es un .npz con 'image' (12, 512, 512) int16, 'aerosol'
    (512, 512) float32 y 'label' (512, 512) uint8 {0,1}. Devuelve (image, label)
    como tensores torch. La aerosol se ignora a propósito.
    """

    REFLECTANCE_SCALE = 10000.0

    def __init__(
        self,
        env_file: str | Path | None = ".env",
        root: str | Path | None = None,
        transform=None,
        normalize: str | bool = "reflectance",
    ) -> None:
        if env_file is None and root is None:
            raise ValueError("Pasá env_file o root.")
        if env_file is not None and root is not None:
            raise ValueError("Pasá solo uno: env_file o root, no ambos.")

        if root is None:
            cfg = load_config(env_file)
            resolved = resolve_root(cfg)
        else:
            resolved = Path(root).expanduser().resolve()

        self.root = resolved
        self.transform = transform
        self.normalize = normalize

        self.samples: list[Path] = []
        for scene in sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("scene")):
            self.samples.extend(sorted(scene.glob("*.npz")))

        if not self.samples:
            raise RuntimeError(f"No se encontraron .npz en {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.normalize is False or self.normalize is None:
            return image.astype(np.float32)
        if self.normalize == "reflectance":
            return (image.astype(np.float32) / self.REFLECTANCE_SCALE)
        raise ValueError(f"normalize no reconocido: {self.normalize!r}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.samples[idx]) as data:
            image = self._normalize(data["image"])
            label = data["label"].astype(np.int64)

        image_t = torch.from_numpy(image)
        label_t = torch.from_numpy(label)

        if self.transform is not None:
            image_t, label_t = self.transform(image_t, label_t)

        return image_t, label_t
