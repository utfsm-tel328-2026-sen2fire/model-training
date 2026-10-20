from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

NormalizeSpec = str | bool | None


class BaseSegmentationDataset(Dataset):
    """Contrato común para datasets de segmentación.

    Una subclase define dos cosas y nada más:
      - `_index()`: qué archivos componen el dataset y en qué orden.
      - `_load_sample(path)`: cómo se convierte un archivo en (image, label).

    Todo lo demás — resolución de rutas, normalización, transformaciones — vive
    acá. Nótese que la clase recibe un `root` ya resuelto: provisionar la fuente
    (NFS, caché local) es responsabilidad de `DatasetSource`/`load_dataset`,
    no del dataset. Eso permite testear con un directorio temporal sin tocar
    variables de entorno.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        normalize: NormalizeSpec = "reflectance",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.normalize = normalize
        self.samples: list[Path] = self._index()

        if not self.samples:
            raise RuntimeError(
                f"{type(self).__name__} no encontró muestras en {self.root}. "
                f"Verificá que la ruta apunte al dataset correcto."
            )

    # -- lo que define cada formato ------------------------------------

    def _index(self) -> list[Path]:
        """Devuelve los archivos del dataset en orden determinista."""
        raise NotImplementedError

    def _load_sample(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Lee un archivo crudo y devuelve (image, label) como arrays numpy."""
        raise NotImplementedError

    # -- comportamiento común ------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def group_of(self, idx: int) -> str:
        """Identificador del grupo al que pertenece la muestra (p.ej. escena).

        Se usa para hacer splits agrupando por estructura de carpetas.
        """
        return self.samples[idx].parent.name

    def groups(self) -> dict[str, list[int]]:
        """Índices agrupados por `group_of`, preservando el orden."""
        out: dict[str, list[int]] = {}
        for i in range(len(self)):
            out.setdefault(self.group_of(i), []).append(i)
        return out

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        if self.normalize is False or self.normalize is None:
            return image.astype(np.float32)
        if self.normalize == "reflectance":
            scale = getattr(self, "REFLECTANCE_SCALE", 10000.0)
            return image.astype(np.float32) / scale
        raise ValueError(
            f"{type(self).__name__}: normalize no reconocido: {self.normalize!r}. "
            f"Usá True/'reflectance' o False."
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._load_sample(self.samples[idx])

        image_t = torch.from_numpy(self.normalize_image(image))
        label_t = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))

        if self.transform is not None:
            image_t, label_t = self.transform(image_t, label_t)

        return image_t, label_t

    def describe(self) -> str:
        return f"{type(self).__name__}(root={self.root}, n={len(self)})"

    def __repr__(self) -> str:
        return self.describe()


def split_by_groups(
    groups: dict[str, list[int]],
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str] = (),
    strict: bool = True,
) -> dict[str, list[int]]:
    """Reparte índices en train/val/test según a qué grupo pertenecen.

    Genérico sobre cualquier dataset que exponga `groups()`.
    """
    def gather(names: Sequence[str]) -> list[int]:
        idx: list[int] = []
        for name in names:
            if name not in groups:
                if strict:
                    raise KeyError(
                        f"grupo {name!r} no encontrado; disponibles: {sorted(groups)}"
                    )
                continue
            idx.extend(groups[name])
        return idx

    out = {"train": gather(train), "val": gather(val)}
    out["test"] = gather(test)
    return out