from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

NFS_SEN2FIRE_RUNS_ROOT_ENV = "NFS_SEN2FIRE_RUNS_ROOT"
USER_ENV = "USER"

DEFAULT_LOCAL_RUNS_ROOT = Path("runs")
DEFAULT_REGISTRY_PATH = Path("experiments.jsonl")


@dataclass(frozen=True)
class RunPaths:
    """Rutas asociadas a un run individual."""
    root: Path                # <runs_root>/<user>/<run_id>/
    configs: Path             # root/configs/
    checkpoints: Path         # root/checkpoints/
    tb_logs: Path             # root/tb_logs/
    stdout: Path              # root/stdout.log
    summary: Path             # root/summary.json


def runs_root() -> Path:
    """Raíz configurable por env para todos los runs. Default: ``./runs``."""
    env_root = os.environ.get(NFS_SEN2FIRE_RUNS_ROOT_ENV)
    return Path(env_root).expanduser() if env_root else DEFAULT_LOCAL_RUNS_ROOT


def username() -> str:
    return os.environ.get(USER_ENV) or os.environ.get("USERNAME") or socket.gethostname()


def git_commit(repo: Path | None = None) -> str:
    """Lee el commit y la rama actuales. Devuelve ``"<commit>[<branch>]"`` o vacío."""
    repo = repo or Path(".")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        branch = "unknown"
    return f"{commit}[{branch}]"


def new_run_id(user: str | None = None) -> str:
    user = user or username()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = int(time.time() * 1000) % 1000
    return f"{user}_{ts}_{suffix:03d}"


def stage_run_dirs(run_id: str) -> RunPaths:
    """Crea los subdirectorios esperados para el run.

    El usuario es responsable de pasar el ``run_id`` generado por :func:`new_run_id`
    (o leído desde una variable de entorno durante ``on_train_start``).
    """
    root = runs_root() / username() / run_id
    paths = RunPaths(
        root=root,
        configs=root / "configs",
        checkpoints=root / "checkpoints",
        tb_logs=root / "tb_logs",
        stdout=root / "stdout.log",
        summary=root / "summary.json",
    )
    for d in (paths.configs, paths.checkpoints, paths.tb_logs):
        d.mkdir(parents=True, exist_ok=True)
    paths.stdout.touch(exist_ok=True)
    return paths


class ExperimentRegistryCallback(pl.Callback):
    """Registra el run en ``experiments.jsonl`` y guarda ``summary.json`` en el NFS.

    Hooks: ``on_train_start`` (stage del árbol), ``on_fit_end`` (status=completed),
    ``on_exception`` (status=crashed con el traceback). Sólo rank-zero (se llama
    desde procesos DDP sin duplicar líneas).

    El callback NO decide dónde caen los checkpoints ni los TB logs: eso lo hace
    el ``TensorBoardLogger(save_dir=...)`` y el ``ModelCheckpoint(dirpath=...)``
    del YAML, que deben apuntar al mismo árbol.
    """

    def __init__(
        self,
        registry_path: str | os.PathLike = DEFAULT_REGISTRY_PATH,
        on_crash_status: str = "crashed",
    ) -> None:
        super().__init__()
        self.registry_path = Path(registry_path)
        self.on_crash_status = on_crash_status
        self._paths: Optional[RunPaths] = None
        self._run_id: Optional[str] = None
        self._started_at: Optional[str] = None
        self._start_time: Optional[float] = None
        self._trainer_global_step_on_finish: Optional[int] = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def paths(self) -> Optional[RunPaths]:
        return self._paths

    @rank_zero_only
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._run_id = new_run_id()
        self._paths = stage_run_dirs(self._run_id)
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._start_time = time.time()

    @rank_zero_only
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._trainer_global_step_on_finish = trainer.global_step
        try:
            self._record(trainer, pl_module, status="completed")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"[experiment-registry] on_fit_end failed: {exc!r}")

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Si vinimos de un fit, ya escribimos completed. Aquí sólo informativo.
        if self._paths is not None and self._run_id is not None:
            pass

    @rank_zero_only
    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        if self._paths is None:
            return
        try:
            self._record(
                trainer,
                pl_module,
                status=self.on_crash_status,
                error=repr(exception),
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"[experiment-registry] on_exception failed: {exc!r}")

    def _record(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        assert self._paths is not None and self._run_id is not None
        duration = time.time() - (self._start_time or time.time())
        ckpt_cb = trainer.checkpoint_callback
        best_metric = getattr(ckpt_cb, "best_model_score", None) if ckpt_cb is not None else None
        best_ckpt_path = getattr(ckpt_cb, "best_model_path", None) if ckpt_cb is not None else None
        best_metric_f: Optional[float] = None
        if best_metric is not None:
            try:
                best_metric_f = float(best_metric.detach().cpu().item())
            except Exception:
                best_metric_f = float(best_metric)

        cbs = trainer.callback_metrics or {}
        def _cb(key: str) -> Optional[float]:
            v = cbs.get(key)
            if v is None:
                return None
            try:
                return float(v.detach().cpu().item())
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return None

        record: dict[str, Any] = {
            "run_id": self._run_id,
            "user": username(),
            "host": socket.gethostname(),
            "created_at": self._started_at,
            "registry_path": str(self.registry_path),
            "git_commit": git_commit(),
            "trainer_max_steps": getattr(trainer, "max_steps", None),
            "trainer_global_step_end": self._trainer_global_step_on_finish or trainer.global_step,
            "trainer_global_rank": trainer.global_rank,
            "status": status,
            "duration_seconds": duration,
            "best_metric": best_metric_f,
            "best_ckpt_path": best_ckpt_path,
            "best_val_iou_fire": _cb("val/iou_fire"),
            "best_val_oa": _cb("val/oa"),
            "best_val_loss": _cb("val/loss"),
            "ckpt_root": str(self._paths.root),
            "checkpoints_dir": str(self._paths.checkpoints),
            "tb_logs_dir": str(self._paths.tb_logs),
            "stdout_log": str(self._paths.stdout),
            "error": error,
        }

        with self._paths.summary.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
