from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

SENTINEL_NAME = ".dataset-cache.json"
SENTINEL_VERSION = 1


@dataclass(frozen=True)
class Manifest:
    """Resumen de un árbol de archivos: cuántos son y.bytes pesan."""

    files: int
    bytes: int


def _scan(root: Path, skip_sentinel: bool = True) -> Manifest:
    """Cuenta archivos y bytes bajo root usando solo metadatos (stat).

    OJO: contra una montura de red esto hace un stat por archivo, así que es
    caro y puede bloquear (autofs). Preferimos nunca llamarlo sobre la fuente
    cuando la caché local ya está validada.
    """
    files = 0
    total = 0
    for path in root.rglob("*"):
        if skip_sentinel and path.name == SENTINEL_NAME:
            continue
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        files += 1
        total += path.stat().st_size
    return Manifest(files=files, bytes=total)


def _sentinel(dst: Path) -> Path:
    return dst / SENTINEL_NAME


def _read_sentinel(dst: Path) -> dict | None:
    path = _sentinel(dst)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != SENTINEL_VERSION:
        return None
    return data


def _write_sentinel(dst: Path, src: Path, manifest: Manifest) -> None:
    payload = {
        "version": SENTINEL_VERSION,
        "src": str(src),
        "files": manifest.files,
        "bytes": manifest.bytes,
    }
    tmp = _sentinel(dst).with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(_sentinel(dst))


def cache_is_usable(src: Path, dst: Path) -> bool:
    """¿La caché local sirve sin necesidad de mirar la fuente?

    Si hay un sello escrito por una clonación previa, basta con comprobar que
    dst sigue teniendo los mismos archivos y bytes que cuando se clonó. No se
    toca `src` en absoluto, así que se puede entrenar con la NFS caída.
    """
    stamp = _read_sentinel(dst)
    if stamp is None or stamp.get("src") != str(src):
        return False
    local = _scan(dst)
    return local.files == stamp.get("files") and local.bytes == stamp.get("bytes")


def cache_matches_source(src: Path, dst: Path) -> bool:
    """Compara origen y caché archivo por archivo.

    Camino caro: hace un stat por archivo en los dos lados. Solo se usa la
    primera vez (o tras cambiar la fuente), nunca en el camino rápido.
    """
    if not src.is_dir() or not dst.is_dir():
        return False
    src_manifest = _scan(src)
    return src_manifest.files > 0 and src_manifest == _scan(dst)


def prepare_cache(src: Path, dst: Path, force: bool = False) -> Path:
    """Clona el árbol src -> dst de forma recursiva e idempotente.

    No conoce el layout del dataset: espeja cualquier estructura de
    directorios. Al terminar escribe un sello en dst para que las próximas
    corridas no necesiten consultar la fuente.

    Devuelve siempre dst.
    """
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()

    if dst.resolve() == src:
        raise RuntimeError(f"La caché y la fuente apuntan al mismo directorio: {dst}")

    if not src.is_dir():
        raise RuntimeError(
            f"La fuente del dataset no existe o no es un directorio: {src}. "
            f"Si esperabas leer de una caché local, revisá que esté completa."
        )

    if force and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"La fuente del dataset no contiene archivos: {src}")

    total_bytes = sum(p.stat().st_size for p in files)
    with tqdm(
        total=total_bytes,
        desc=f"copiando {src.name} -> {dst}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True,
    ) as bar:
        for path in files:
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            bar.update(path.stat().st_size)

    _write_sentinel(dst, src, Manifest(files=len(files), bytes=total_bytes))
    return dst


class DatasetSource:
    """De dónde salen los datos y desde dónde hay que leerlos realmente.

    Dos modos:
      - sin caché: la raíz efectiva es el origen (p.ej. NFS, leído cada epoch).
      - con caché: el origen se clona una sola vez a un directorio local y
        toda la sesión lee desde ahí. Cuando el sello está vigente no se toca
        la montura de red.

    Separar esto del formato del dataset es lo que permite que la misma lógica
    de caché sirva para cualquier dataset en cualquier formato.
    """

    def __init__(self, src: Path | str, cache_dir: Path | str | None = None) -> None:
        self.src = Path(src).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None

    @property
    def is_cached(self) -> bool:
        return self.cache_dir is not None

    def materialize(self, force: bool = False) -> Path:
        """Devuelve la ruta desde la que hay que leer, clonando si hace falta."""
        if self.cache_dir is None:
            if not self.src.is_dir():
                raise RuntimeError(
                    f"El dataset no existe o no es un directorio: {self.src}"
                )
            return self.src

        if self.cache_dir == self.src:
            raise RuntimeError(
                f"La caché y la fuente apuntan al mismo directorio: {self.src}. "
                f"Indicá una caché distinta o quitá la variable de caché."
            )

        # Camino rápido: sello válido -> ni stat a la fuente de red.
        if not force and cache_is_usable(self.src, self.cache_dir):
            return self.cache_dir

        # La caché existe pero no hay sello (p.ej. clonada a mano). Verificamos
        # contra la fuente una vez y escribimos el sello para las próximas.
        if not force and self.cache_dir.is_dir() and cache_matches_source(
            self.src, self.cache_dir
        ):
            _write_sentinel(self.cache_dir, self.src, _scan(self.cache_dir))
            return self.cache_dir

        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        return prepare_cache(self.src, self.cache_dir, force=force)

    def __repr__(self) -> str:
        return f"DatasetSource(src={self.src}, cache={self.cache_dir})"