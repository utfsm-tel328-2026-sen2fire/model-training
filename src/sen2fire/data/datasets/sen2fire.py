from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..base import BaseSegmentationDataset, split_by_groups

# Escenas del split por defecto del paper de Sen2Fire.
DEFAULT_TRAIN = ("scene1", "scene2")
DEFAULT_VAL = ("scene3",)
DEFAULT_TEST = ("scene4",)


class Sen2FireNpzDataset(BaseSegmentationDataset):
    """Dataset Sen2Fire en formato .npz.

    Espera el layout:
        <root>/scene1/*.npz
        <root>/scene2/*.npz
        ...

    Cada .npz contiene:
        'image'    (12, 512, 512) int16   reflectancia cruda
        'aerosol'  (512, 512)     float32 ignorado a propósito
        'label'    (512, 512)     uint8   {0, 1}

    Devuelve (image float32 normalizada, label int64 en {0, 1}).

    Esta clase solo sabe leer .npz de Sen2Fire. De dónde vienen los archivos
    (NFS o caché local) lo resuelve `load_dataset()` antes de construirla.
    """

    REFLECTANCE_SCALE = 10000.0

    def _index(self) -> list[Path]:
        scenes = sorted(
            p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("scene")
        )
        samples: list[Path] = []
        for scene in scenes:
            samples.extend(sorted(scene.glob("*.npz")))
        return samples

    def _load_sample(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path) as data:
            image = data["image"]
            label = data["label"]
        return image, label

    def describe(self) -> str:
        n_scenes = len({p.parent.name for p in self.samples})
        return (
            f"Sen2FireNpzDataset(root={self.root}, n={len(self)}, "
            f"scenes={n_scenes})"
        )


def get_scene_split(
    ds: BaseSegmentationDataset,
    train: Sequence[str] = DEFAULT_TRAIN,
    val: Sequence[str] = DEFAULT_VAL,
    test: Sequence[str] = DEFAULT_TEST,
    strict: bool = True,
) -> dict[str, list[int]]:
    """Split por escena: train=scene1+scene2, val=scene3, test=scene4.

    Recibe el dataset (no una lista de paths) para no depender de que
    `samples` sea un atributo con esa forma exacta.
    """
    return split_by_groups(ds.groups(), train, val, test, strict=strict)