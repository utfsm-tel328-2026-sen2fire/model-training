from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_path: Path
    cache_path: Path | None


def load_config(env_file: str | os.PathLike | None = ".env") -> Config:
    from dotenv import load_dotenv

    if env_file is not None:
        load_dotenv(env_file, override=False)

    data = os.environ.get("SEN2FIRE_ROOT")
    if not data:
        raise RuntimeError(
            "SEN2FIRE_ROOT no está definido. Configúralo en el .env o pásalo "
            "como variable de entorno (apunta a la carpeta que contiene "
            "scene1/, scene2/, ...)."
        )

    data_path = Path(data).expanduser().resolve()
    if not data_path.is_dir():
        raise RuntimeError(f"SEN2FIRE_ROOT no existe o no es un directorio: {data_path}")
    if not any(data_path.glob("scene*")):
        raise RuntimeError(
            f"SEN2FIRE_ROOT no contiene ninguna subcarpeta scene*: {data_path}"
        )

    cache_raw = os.environ.get("SEN2FIRE_LOCAL_CACHE")
    cache_path: Path | None = None
    if cache_raw:
        cache_path = Path(cache_raw).expanduser().resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    return Config(data_path=data_path, cache_path=cache_path)
