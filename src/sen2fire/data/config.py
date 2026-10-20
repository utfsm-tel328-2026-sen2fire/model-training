from __future__ import annotations

import os
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENV_FILE = ".env"
DEFAULT_DATASET = "sen2fire"


def env_prefix(name: str) -> str:
    """'sen2fire' -> 'SEN2FIRE'. El prefijo con el que se nombran sus variables."""
    return name.upper().replace("-", "_")


def env_lookup(env_file: str | os.PathLike | None = DEFAULT_ENV_FILE) -> ChainMap:
    """Capas de configuración: el entorno real gana sobre el .env.

    Devolver un ChainMap en vez de llamar a load_dotenv() evita mutar os.environ
    como efecto colateral, que hace imposible testear distintos .env en la
    misma sesión y contamina cualquier código que se ejecute después.
    """
    from dotenv import dotenv_values

    file_values: dict[str, str] = {}
    if env_file is not None:
        path = Path(env_file)
        if path.is_file():
            file_values = {k: v for k, v in dotenv_values(path).items() if v is not None}

    return ChainMap(os.environ, file_values)


@dataclass(frozen=True)
class SourceConfig:
    """Dónde vive un dataset y si debe clonarse a local antes de leerlo."""

    root: Path
    cache_dir: Path | None = None

    @property
    def wants_local_copy(self) -> bool:
        return self.cache_dir is not None


def get_source_config(
    name: str, env_file: str | os.PathLike | None = DEFAULT_ENV_FILE
) -> SourceConfig:
    """Lee la config de un dataset por convención: <PREFIX>_ROOT, <PREFIX>_LOCAL_CACHE.

    `root` es obligatorio; `cache_dir` es opcional y, si está, activa el clonado
    a local una única vez por sesión de entrenamiento.
    """
    env = env_lookup(env_file)
    prefix = env_prefix(name)

    root_raw = env.get(f"{prefix}_ROOT")
    if not root_raw:
        raise RuntimeError(
            f"{prefix}_ROOT no está definido. Configuralo en el .env o pasalo como "
            f"variable de entorno (apunta a la carpeta que contiene los datos "
            f"del dataset {name!r})."
        )

    root = Path(root_raw).expanduser().resolve()

    cache_raw = env.get(f"{prefix}_LOCAL_CACHE")
    cache_dir = Path(cache_raw).expanduser().resolve() if cache_raw else None

    # Nota: no validamos que root exista acá a propósito. Si hay caché local
    # completa, se puede entrenar aunque la montura NFS no esté disponible.
    # `DatasetSource.materialize()` valida solo cuando realmente necesita leer
    # del origen.
    return SourceConfig(root=root, cache_dir=cache_dir)