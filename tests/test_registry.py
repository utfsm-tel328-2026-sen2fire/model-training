"""Tests del callback de registro de experimentos."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytorch_lightning as pl
import pytest

from sen2fire.training.registry import (
    DEFAULT_LOCAL_RUNS_ROOT,
    ExperimentRegistryCallback,
    RunPaths,
    git_commit,
    new_run_id,
    runs_root,
    stage_run_dirs,
    username,
)


def _trainer_mock() -> MagicMock:
    """Mock minimal del Trainer que satisface lo que `_record` necesita."""
    m = MagicMock()
    m.checkpoint_callback = MagicMock()
    m.checkpoint_callback.best_model_score = 0.42
    m.checkpoint_callback.best_model_path = "/tmp/ckpt.ckpt"
    m.callback_metrics = {"val/iou_fire": 0.42, "val/oa": 0.95}
    m.global_step = 5
    m.global_rank = 0
    m.max_steps = 5000
    return m


def test_runs_root_respects_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NFS_SEN2FIRE_RUNS_ROOT", str(tmp_path / "nfs"))
    root = runs_root()
    assert root == tmp_path / "nfs"


def test_runs_root_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NFS_SEN2FIRE_RUNS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert runs_root() == DEFAULT_LOCAL_RUNS_ROOT


def test_stage_run_dirs_creates_tree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NFS_SEN2FIRE_RUNS_ROOT", str(tmp_path / "nfs"))
    monkeypatch.setenv("USER", "khaen")
    rid = new_run_id()
    paths = stage_run_dirs(rid)
    assert isinstance(paths, RunPaths)
    assert paths.root.is_dir()
    assert paths.configs.is_dir()
    assert paths.checkpoints.is_dir()
    assert paths.tb_logs.is_dir()
    assert paths.stdout.is_file()


def test_callback_writes_registry_line(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NFS_SEN2FIRE_RUNS_ROOT", str(tmp_path / "nfs"))
    monkeypatch.setenv("USER", "khaen")
    registry = tmp_path / "experiments.jsonl"
    cb = ExperimentRegistryCallback(registry_path=registry)
    trainer = _trainer_mock()

    cb.on_train_start(trainer, None)
    cb._record(trainer, None, status="completed")

    assert cb.run_id is not None
    assert registry.is_file()
    lines = registry.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "completed"
    assert record["run_id"] == cb.run_id
    assert record["user"] == "khaen"
    assert record["best_val_iou_fire"] == 0.42
    assert (Path(record["ckpt_root"]) / "summary.json").is_file()


def test_callback_writes_crashed_on_exception(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NFS_SEN2FIRE_RUNS_ROOT", str(tmp_path / "nfs"))
    monkeypatch.setenv("USER", "khaen")
    registry = tmp_path / "experiments.jsonl"
    cb = ExperimentRegistryCallback(registry_path=registry)
    trainer = _trainer_mock()

    cb.on_train_start(trainer, None)
    cb._record(trainer, None, status="crashed", error="RuntimeError('boom')")

    lines = registry.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "crashed"
    assert "boom" in record["error"]


def test_git_commit_returns_string() -> None:
    commit = git_commit()
    # En repo con commit devuelve "<sha>[<branch>]"; sin git devuelve "".
    assert isinstance(commit, str)
    if commit:
        assert "[" in commit


def test_username_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("USER", "khaen")
    monkeypatch.delenv("USERNAME", raising=False)
    assert username() == "khaen"
