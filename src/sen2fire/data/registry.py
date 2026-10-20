from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .base import BaseSegmentationDataset
from .config import DEFAULT_DATASET, DEFAULT_ENV_FILE, get_source_config
from .datasets.sen2fire import Sen2FireNpzDataset
from .source import DatasetSource


@dataclass(frozen=True)
class DatasetSpec:
    """Registro de un dataset: qué clase lo implementa y cómo se llama en el .env."""

    name: str
    dataset_cls: type[BaseSegmentationDataset]


_REGISTRY: dict[str, DatasetSpec] = {}


def register(name: str, dataset_cls: type[BaseSegmentationDataset]) -> DatasetSpec:
    spec = DatasetSpec(name=name, dataset_cls=dataset_cls)
    _REGISTRY[name] = spec
    return spec


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


def load_dataset(
    name: str | None = None,
    env_file: str | os.PathLike | None = DEFAULT_ENV_FILE,
    **kwargs: Any,
) -> BaseSegmentationDataset:
    """Construye un dataset resolviendo su fuente de datos.

    El flujo es siempre el mismo, sin importar el formato:
        1. Se lee la config por convención (`<PREFIX>_ROOT`,
           `<PREFIX>_LOCAL_CACHE`) desde el entorno / .env.
        2. `DatasetSource` decide si hay que leer del origen o de una copia local
           (clonando una sola vez si hace falta).
        3. Se instancia la clase del dataset con un `root` ya resuelto.

    Parámetros:
        name: dataset a cargar. Si es None se lee `DATASET` del entorno, y si
            tampoco está se usa "sen2fire".
        kwargs: se pasan a la clase del dataset (transform, normalize, ...).

    Agregar un dataset nuevo = registrarlo acá. No se toca train.py ni main.py.
    """
    if name is None:
        name = os.environ.get("DATASET", DEFAULT_DATASET)

    spec = _REGISTRY.get(name)
    if spec is None:
        raise KeyError(
            f"dataset desconocido: {name!r}. Disponibles: {available_datasets()}"
        )

    cfg = get_source_config(name, env_file=env_file)
    source = DatasetSource(cfg.root, cfg.cache_dir)
    root = source.materialize()

    return spec.dataset_cls(root=root, **kwargs)


register("sen2fire", Sen2FireNpzDataset)