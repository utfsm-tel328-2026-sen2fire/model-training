from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import pytorch_lightning as L
from torch.utils.data import DataLoader, Subset

from .base import BaseSegmentationDataset, split_by_groups
from .config import DEFAULT_DATASET, DEFAULT_ENV_FILE, get_source_config
from .datasets.sen2fire import get_scene_split
from .registry import load_dataset
from .source import DatasetSource

NormalizeSpec = str | bool | None


class SegmentationDataModule(L.LightningDataModule):
    """LightningDataModule para segmentación.

    Encapsula fuente (NFS / caché local), instanciación del dataset y split por
    escena. Diseñado para ser usable sin Trainer — útil para `main.py` y los
    tests — y para integrarse con `Trainer` cuando llegue el `LightningModule`.

    Por qué se cachea `self.ds` en `_build()`: `setup(stage)` es invocado una vez
    por stage (fit, test, predict). En Lightning con `strategy="ddp"` también se
    re-invoca por proceso. Sin cachear, cada llamada relistaría el árbol de
    archivos (`_index()` hace glob sobre todos los .npz), que contra NFS es
    lento. Con cache, la segunda llamada es un no-op.

    Args:
        name: nombre en el registro de datasets (`os.environ["DATASET"]` si
            es None). Por defecto, "sen2fire".
        env_file: ruta al `.env` con `<PREFIX>_ROOT` y `<PREFIX>_LOCAL_CACHE`.
            `None` lee solo `os.environ`.
        batch_size, num_workers, pin_memory: parámetros estándar del DataLoader.
        transform: callable opcional `(image_t, label_t) -> (image_t, label_t)`,
            recibe tensores (no numpy). Ver `BaseSegmentationDataset`.
        normalize: "reflectance"/True o False. Ver `BaseSegmentationDataset`.
        splits: tupla `(train, val, test)` con los nombres de grupo. Por
            defecto se usa `get_scene_split()` de Sen2Fire.
        limit_train, limit_val: recortan train/val a los primeros N índices
            (smoke tests).
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        env_file: str | os.PathLike | None = DEFAULT_ENV_FILE,
        batch_size: int = 2,
        num_workers: int = 2,
        pin_memory: bool = False,
        transform: Callable[..., Any] | None = None,
        normalize: NormalizeSpec = "reflectance",
        splits: tuple[Sequence[str], Sequence[str], Sequence[str]] | None = None,
        limit_train: int | None = None,
        limit_val: int | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.env_file = env_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform
        self.normalize = normalize
        self.splits = splits
        self.limit_train = limit_train
        self.limit_val = limit_val

        self.ds: BaseSegmentationDataset | None = None
        self._split: dict[str, list[int]] | None = None

    # --- helper -------------------------------------------------------

    def _resolved_name(self) -> str:
        return self.name or os.environ.get("DATASET", DEFAULT_DATASET)

    def _build(self) -> None:
        """Construye self.ds y self._split exactamente una vez por instancia."""
        if self.ds is not None:
            return

        ds_kwargs: dict[str, Any] = {"normalize": self.normalize}
        if self.transform is not None:
            ds_kwargs["transform"] = self.transform

        self.ds = load_dataset(self._resolved_name(), env_file=self.env_file, **ds_kwargs)

        if self.splits is None:
            self._split = get_scene_split(self.ds)
        else:
            self._split = split_by_groups(
                self.ds.groups(),
                *self.splits,
                strict=True,
            )

    # --- hooks de LightningDataModule ----------------------------------

    def prepare_data(self) -> None:
        """Materializa la fuente (NFS + caché local opcional).

        Idempotente: si la copia local ya tiene un sello vigente, no se toca
        la montura de red. Si no hay caché, devuelve el root original.

        Lightning invoca esto una sola vez por nodo en el proceso padre. Sin
        `Trainer`, llamarlo a mano antes de `setup("fit")`.
        """
        cfg = get_source_config(self._resolved_name(), env_file=self.env_file)
        DatasetSource(cfg.root, cfg.cache_dir).materialize()

    def setup(self, stage: str | None) -> None:
        """Instancia el dataset y arma los `Subset` por stage, cacheando.

        - stage in {"fit", "validate", None}  -> construye `train_dataset` y
          `val_dataset`. Los flags `limit_*` se aplican a los índices antes de
          el Subset.
        - stage in {"test", None}  -> construye `test_dataset`.
        - stage in {"predict"}     -> construye `predict_dataset` con todos los
          índices.
        """
        self._build()
        assert self.ds is not None and self._split is not None

        fit_like = stage in ("fit", "validate", None)
        if fit_like:
            train_idx = list(self._split["train"])
            val_idx = list(self._split["val"])
            if self.limit_train is not None:
                train_idx = train_idx[: self.limit_train]
            if self.limit_val is not None:
                val_idx = val_idx[: self.limit_val]
            self.train_dataset = Subset(self.ds, train_idx)
            self.val_dataset = Subset(self.ds, val_idx)

        if stage in ("test", None):
            self.test_dataset = Subset(self.ds, list(self._split["test"]))

        if stage == "predict":
            all_idx = (
                list(self._split["train"])
                + list(self._split["val"])
                + list(self._split["test"])
            )
            self.predict_dataset = Subset(self.ds, all_idx)

    # --- dataloaders ---------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    # --- misc ----------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self.ds) if self.ds is not None else "?"
        return (
            f"SegmentationDataModule(name={self._resolved_name()!r}, "
            f"batch_size={self.batch_size}, num_workers={self.num_workers}, "
            f"n={n})"
        )
