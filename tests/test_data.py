"""Tests del flujo de carga: fuente, caché, formato y split.

Corre sin red y sin .env: todo apunta a directorios temporales.
Ejecutar con: python -m pytest tests/ -v   (o: python tests/test_data.py)
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from sen2fire.data import DatasetSource, load_dataset
from sen2fire.data.datasets.sen2fire import Sen2FireNpzDataset, get_scene_split
from sen2fire.data.source import SENTINEL_NAME, cache_is_usable, prepare_cache

from make_fake_data import make_fake_scene


def _fixture(tmp: Path, per_scene: int = 4, size: int = 32) -> Path:
    for s in range(1, 5):
        make_fake_scene(tmp / f"scene{s}", per_scene, size, seed=s)
    return tmp


def test_clon_a_cache_y_es_idempotente(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src", per_scene=3)
    cache = tmp_path / "cache"

    first = DatasetSource(src, cache).materialize()
    assert first == cache.resolve()
    assert len(list(cache.rglob("*.npz"))) == 12
    assert (cache / SENTINEL_NAME).is_file()

    # Segunda pasada: no debe volver a copiar ni a tocar la fuente.
    stamp = json.loads((cache / SENTINEL_NAME).read_text())
    second = DatasetSource(src, cache).materialize()
    assert second == cache.resolve()
    assert json.loads((cache / SENTINEL_NAME).read_text()) == stamp


def test_cache_incompleta_se_reclona(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src", per_scene=3)
    cache = tmp_path / "cache"
    DatasetSource(src, cache).materialize()

    # Borramos un archivo: la caché ya no sirve y debe re-clonarse.
    victim = next(cache.rglob("*.npz"))
    victim.unlink()
    assert not cache_is_usable(src, cache.resolve())

    DatasetSource(src, cache).materialize()
    assert len(list(cache.rglob("*.npz"))) == 12
    assert cache_is_usable(src, cache.resolve())


def test_cache_valida_sin_tocar_la_fuente(tmp_path: Path) -> None:
    """El camino rápido no debe hacer stat sobre la fuente.

    Es lo que permite entrenar con la NFS caída.
    """
    src = _fixture(tmp_path / "src", per_scene=3)
    cache = tmp_path / "cache"
    DatasetSource(src, cache).materialize()

    src_renamed = tmp_path / "src-fuera-de-linea"
    src.rename(src_renamed)

    assert cache_is_usable(src, cache.resolve())
    resolved = DatasetSource(src, cache).materialize()
    assert resolved == cache.resolve()


def test_cache_con_otra_fuente_se_reclona(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src-a", per_scene=3)
    cache = tmp_path / "cache"
    DatasetSource(src, cache).materialize()

    src_b = _fixture(tmp_path / "src-b", per_scene=5)
    DatasetSource(src_b, cache).materialize()

    assert len(list(cache.rglob("*.npz"))) == 20


def test_force_reclona(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src", per_scene=3)
    cache = tmp_path / "cache"
    DatasetSource(src, cache).materialize()
    (cache / "scene1" / "basura.npz").write_bytes(b"x")

    prepare_cache(src, cache, force=True)
    assert not (cache / "scene1" / "basura.npz").exists()


def test_source_y_cache_iguales_es_error(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src", per_scene=2)
    try:
        DatasetSource(src, src).materialize()
    except RuntimeError as exc:
        assert "mismo directorio" in str(exc)
    else:
        raise AssertionError("debería fallar")


def test_formato_y_normalizacion(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src", per_scene=2, size=32)
    ds = Sen2FireNpzDataset(root=src)

    assert len(ds) == 8
    assert ds.describe().startswith("Sen2FireNpzDataset")

    image, label = ds[0]
    assert image.shape == (12, 32, 32)
    assert label.shape == (32, 32)
    assert image.dtype == torch.float32
    assert label.dtype == torch.int64
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0

    raw = Sen2FireNpzDataset(root=src, normalize=False)
    assert float(raw[0][0].max()) > 1.0


def test_dataset_acepta_root_directo_sin_env(tmp_path: Path) -> None:
    """El dataset no debe leer el .env: se le pasa un root ya resuelto."""
    src = _fixture(tmp_path / "src", per_scene=2)
    env_backup = {k: os.environ.get(k) for k in ("SEN2FIRE_ROOT", "SEN2FIRE_LOCAL_CACHE")}
    for k in env_backup:
        os.environ.pop(k, None)
    try:
        ds = Sen2FireNpzDataset(root=src)
        assert len(ds) == 8
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v


def test_split_por_escena(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src", per_scene=5)
    ds = Sen2FireNpzDataset(root=src)

    split = get_scene_split(ds)
    assert len(split["train"]) == 10
    assert len(split["val"]) == 5
    assert len(split["test"]) == 5
    assert not set(split["train"]) & set(split["val"])

    # Escena inexistente debe fallar claro.
    try:
        get_scene_split(ds, train=("scene9",), val=("scene3",))
    except KeyError as exc:
        assert "scene9" in str(exc)
    else:
        raise AssertionError("debería fallar")


def test_load_dataset_desde_env(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src", per_scene=3)
    cache = tmp_path / "cache"
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        f"SEN2FIRE_ROOT={src}\nSEN2FIRE_LOCAL_CACHE={cache}\n"
    )

    ds = load_dataset("sen2fire", env_file=env_file)
    assert ds.root == cache.resolve()
    assert len(ds) == 12

    # Segunda llamada: usa la caché ya clonada.
    again = load_dataset("sen2fire", env_file=env_file)
    assert again.root == cache.resolve()


def test_load_dataset_sin_cache_usa_el_origen(tmp_path: Path) -> None:
    src = _fixture(tmp_path / "src", per_scene=3)
    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={src}\n")

    ds = load_dataset("sen2fire", env_file=env_file)
    assert ds.root == src.resolve()


def test_dataset_desconocido(tmp_path: Path) -> None:
    try:
        load_dataset("no-existe", env_file=None)
    except KeyError as exc:
        assert "no-existe" in str(exc)
    else:
        raise AssertionError("debería fallar")


def _main() -> int:
    failures = 0
    tmp = Path(tempfile.mkdtemp(prefix="sen2fire-tests-"))
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for fn in tests:
            case_dir = tmp / fn.__name__
            case_dir.mkdir(parents=True)
            try:
                fn(case_dir)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
                import traceback

                traceback.print_exc()
            else:
                print(f"ok   {fn.__name__}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failures}/{len(tests)} tests pasaron")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())