"""Tests del flujo de carga: fuente, caché, formato y split.

Corren sin red, sin GPU y sin el `.env` real: todo apunta a directorios
temporales con datos sintéticos.

Ejecutar con: uv run pytest
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sen2fire.data import DatasetSource, SegmentationDataModule, load_dataset
from sen2fire.data.base import BaseSegmentationDataset
from sen2fire.data.datasets.sen2fire import Sen2FireNpzDataset, get_scene_split
from sen2fire.data.source import SENTINEL_NAME, cache_is_usable, prepare_cache

from conftest import make_scene


# -- fuente y caché ----------------------------------------------------


def test_clon_a_cache_y_es_idempotente(fake_src: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"

    first = DatasetSource(fake_src, cache).materialize()
    assert first == cache.resolve()
    assert len(list(cache.rglob("*.npz"))) == 16
    assert (cache / SENTINEL_NAME).is_file()

    # Segunda pasada: no debe volver a copiar ni a tocar la fuente.
    stamp = json.loads((cache / SENTINEL_NAME).read_text())
    second = DatasetSource(fake_src, cache).materialize()
    assert second == cache.resolve()
    assert json.loads((cache / SENTINEL_NAME).read_text()) == stamp


def test_cache_incompleta_se_reclona(fake_src: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    DatasetSource(fake_src, cache).materialize()

    # Borramos un archivo: la caché ya no sirve y debe re-clonarse.
    next(cache.rglob("*.npz")).unlink()
    assert not cache_is_usable(fake_src, cache.resolve())

    DatasetSource(fake_src, cache).materialize()
    assert len(list(cache.rglob("*.npz"))) == 16
    assert cache_is_usable(fake_src, cache.resolve())


def test_cache_valida_sin_tocar_la_fuente(fake_src: Path, tmp_path: Path) -> None:
    """El camino rápido no debe hacer stat sobre la fuente.

    Es lo que permite entrenar con la NFS caída.
    """
    cache = tmp_path / "cache"
    DatasetSource(fake_src, cache).materialize()

    fake_src.rename(tmp_path / "src-fuera-de-linea")

    assert cache_is_usable(fake_src, cache.resolve())
    assert DatasetSource(fake_src, cache).materialize() == cache.resolve()


def test_cache_con_otra_fuente_se_reclona(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    DatasetSource(make_scene(tmp_path / "src-a", per_scene=3), cache).materialize()

    src_b = make_scene(tmp_path / "src-b", per_scene=5)
    DatasetSource(src_b, cache).materialize()

    assert len(list(cache.rglob("*.npz"))) == 20


def test_force_reclona(fake_src: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    DatasetSource(fake_src, cache).materialize()
    (cache / "scene1" / "basura.npz").write_bytes(b"x")

    prepare_cache(fake_src, cache, force=True)
    assert not (cache / "scene1" / "basura.npz").exists()


def test_source_y_cache_iguales_es_error(fake_src: Path) -> None:
    with pytest.raises(RuntimeError, match="mismo directorio"):
        DatasetSource(fake_src, fake_src).materialize()


# -- formato ------------------------------------------------------------


def test_formato_y_normalizacion(fake_src: Path) -> None:
    ds = Sen2FireNpzDataset(root=fake_src)

    assert len(ds) == 16
    assert ds.describe().startswith("Sen2FireNpzDataset")

    image, label = ds[0]
    assert image.shape == (12, 32, 32)
    assert label.shape == (32, 32)
    assert image.dtype == torch.float32
    assert label.dtype == torch.int64
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0

    raw = Sen2FireNpzDataset(root=fake_src, normalize=False)
    assert float(raw[0][0].max()) > 1.0


def test_root_explicito_ignora_el_env(fake_src: Path, tmp_path: Path, monkeypatch) -> None:
    """La clase recibe un root ya resuelto: no debe leer el entorno."""
    monkeypatch.setenv("SEN2FIRE_ROOT", str(tmp_path / "no-existe"))
    monkeypatch.setenv("SEN2FIRE_LOCAL_CACHE", str(tmp_path / "tampoco-existe"))

    ds = Sen2FireNpzDataset(root=fake_src)
    assert ds.root == fake_src.resolve()
    assert len(ds) == 16


# -- split --------------------------------------------------------------


def test_split_por_escena(fake_src: Path) -> None:
    ds = Sen2FireNpzDataset(root=fake_src)

    split = get_scene_split(ds)
    assert len(split["train"]) == 8
    assert len(split["val"]) == 4
    assert len(split["test"]) == 4
    assert not set(split["train"]) & set(split["val"])


def test_split_con_escena_inexistente(fake_src: Path) -> None:
    ds = Sen2FireNpzDataset(root=fake_src)
    with pytest.raises(KeyError, match="scene9"):
        get_scene_split(ds, train=("scene9",), val=("scene3",))


# -- carga desde el entorno ---------------------------------------------


def test_load_dataset_desde_env(fake_src: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        f"SEN2FIRE_ROOT={fake_src}\nSEN2FIRE_LOCAL_CACHE={cache}\n"
    )

    ds = load_dataset("sen2fire", env_file=env_file)
    assert ds.root == cache.resolve()
    assert len(ds) == 16

    # Segunda llamada: usa la caché ya clonada.
    assert load_dataset("sen2fire", env_file=env_file).root == cache.resolve()


def test_load_dataset_sin_cache_usa_el_origen(fake_src: Path, tmp_path: Path) -> None:
    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={fake_src}\n")

    ds = load_dataset("sen2fire", env_file=env_file)
    assert ds.root == fake_src.resolve()


def test_env_no_contamina_entre_tests(fake_src: Path, tmp_path: Path) -> None:
    """get_source_config no debe mutar os.environ como efecto colateral."""
    import os

    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={fake_src}\n")

    before = dict(os.environ)
    load_dataset("sen2fire", env_file=env_file)
    assert dict(os.environ) == before


def test_dataset_desconocido() -> None:
    with pytest.raises(KeyError, match="no-existe"):
        load_dataset("no-existe", env_file=None)


# -- LightningDataModule ----------------------------------------------


def test_datamodule_setup_split_por_escena(fake_src: Path, tmp_path: Path) -> None:
    """setup('fit') arma los Subsets train/val con las cardinalidades correctas."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={fake_src}\n")

    dm = SegmentationDataModule(env_file=env_file, batch_size=4, num_workers=0)
    dm.prepare_data()
    dm.setup("fit")

    assert isinstance(dm.ds, Sen2FireNpzDataset)
    assert len(dm.ds) == 16
    assert len(dm.train_dataset) == 8
    assert len(dm.val_dataset) == 4
    assert not set(dm.train_dataset.indices) & set(dm.val_dataset.indices)


def test_datamodule_dataloader_shapes_y_dtypes(fake_src: Path, tmp_path: Path) -> None:
    """train_dataloader() entrega batches colados correctos."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={fake_src}\n")

    dm = SegmentationDataModule(env_file=env_file, batch_size=2, num_workers=0)
    dm.prepare_data()
    dm.setup("fit")

    img, lbl = next(iter(dm.train_dataloader()))
    assert img.dtype == torch.float32
    assert lbl.dtype == torch.int64
    assert img.ndim == 4 and lbl.ndim == 3
    assert img.shape[1:] == (12, 32, 32)
    assert lbl.shape[1:] == (32, 32)
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


def test_datamodule_prepare_data_idempotente(fake_src: Path, tmp_path: Path) -> None:
    """prepare_data() puede invocarse desde varias instancias sin clonar de más."""
    cache = tmp_path / "cache"
    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={fake_src}\nSEN2FIRE_LOCAL_CACHE={cache}\n")

    SegmentationDataModule(env_file=env_file).prepare_data()
    first_sentinel = json.loads((cache / SENTINEL_NAME).read_text())

    SegmentationDataModule(env_file=env_file).prepare_data()
    second_sentinel = json.loads((cache / SENTINEL_NAME).read_text())

    assert first_sentinel == second_sentinel
    assert len(list(cache.rglob("*.npz"))) == 16


def test_datamodule_setup_no_reinstancia_dataset(fake_src: Path, tmp_path: Path) -> None:
    """Las llamadas sucesivas a setup(stage) no reconstruyen self.ds.

    Convención de Lightning: `prepare_data()` no instancia el dataset
    (solo materializa la fuente), y `setup(stage)` cachea en la primera
    invocación para no relistar la NFS en `setup("validate")`/`("test")`/.
    """
    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={fake_src}\n")

    dm = SegmentationDataModule(env_file=env_file, batch_size=2, num_workers=0)

    constructed: list[int] = []
    orig_init = BaseSegmentationDataset.__init__

    def spy(self, *args, **kwargs):
        constructed.append(1)
        return orig_init(self, *args, **kwargs)

    with patch.object(BaseSegmentationDataset, "__init__", spy):
        dm.prepare_data()
        dm.setup("fit")
        dm.train_dataloader()
        dm.setup("test")
        dm.test_dataloader()

    # Solo una construcción: prepare_data no instancia nada, y las dos
    # llamadas a setup usan el cache self.ds.
    assert len(constructed) == 1


def test_datamodule_sin_cache_usa_el_origen(fake_src: Path, tmp_path: Path) -> None:
    """Sin SEN2FIRE_LOCAL_CACHE, dm.ds.root apunta al root original."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={fake_src}\n")

    dm = SegmentationDataModule(env_file=env_file, batch_size=2, num_workers=0)
    dm.prepare_data()
    dm.setup("fit")

    assert dm.ds.root == fake_src.resolve()
    assert len(dm.train_dataset) == 8


def test_datamodule_limit_train_y_val(fake_src: Path, tmp_path: Path) -> None:
    """Los flags limit_train/limit_val recortan los Subsets como en train.py."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(f"SEN2FIRE_ROOT={fake_src}\n")

    dm = SegmentationDataModule(
        env_file=env_file, batch_size=1, num_workers=0,
        limit_train=3, limit_val=1,
    )
    dm.prepare_data()
    dm.setup("fit")

    assert len(dm.train_dataset) == 3
    assert len(dm.val_dataset) == 1