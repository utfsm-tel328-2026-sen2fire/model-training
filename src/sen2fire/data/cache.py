from __future__ import annotations

import shutil
from pathlib import Path

from tqdm import tqdm

from .config import Config


def _scene_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("scene"))


def _scene_complete(src: Path, dst: Path) -> bool:
    if not dst.is_dir():
        return False
    src_files = [p for p in src.iterdir() if p.is_file()]
    dst_files = [p for p in dst.iterdir() if p.is_file()]
    return len(src_files) == len(dst_files) and len(src_files) > 0


def prepare_cache(src: Path, dst: Path, force: bool = False) -> Path:
    """Clona src/scene* a dst/scene* si hace falta. Idempotente.

    Compara por cantidad de archivos por escena: si ya coincide, no copia.
    Devuelve siempre dst.
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    scenes = _scene_dirs(src)
    if not scenes:
        raise RuntimeError(f"No hay subcarpetas scene* en {src}")

    for scene in scenes:
        target = dst / scene.name
        if force and target.exists():
            shutil.rmtree(target)
        if _scene_complete(scene, target):
            continue
        target.mkdir(parents=True, exist_ok=True)
        files = [p for p in scene.iterdir() if p.is_file()]
        with tqdm(
            total=len(files),
            desc=f"copiando {scene.name}",
            unit="archivos",
            leave=True,
        ) as bar:
            for f in files:
                shutil.copy2(f, target / f.name)
                bar.update(1)

    return dst


def resolve_root(cfg: Config) -> Path:
    """Si hay cache_path configurado, prepara el caché y devuelve esa ruta.
    Si no, devuelve la ruta original del dataset.
    """
    if cfg.cache_path is None:
        return cfg.data_path
    return prepare_cache(cfg.data_path, cfg.cache_path)
