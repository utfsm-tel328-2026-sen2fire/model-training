"""Reproducción de runs a partir del JSONL ``experiments.jsonl``.

Hay tres lecturas posibles en el día a día:
  - saber qué corrieron los demás (lista, mejores runs);
  - cargar los pesos de un run concreto;
  - repetir exactamente ese run en la versión de código que estaba commiteada.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

REGISTRY_FILE = Path("experiments.jsonl")


def _load(registry: Path = REGISTRY_FILE) -> list[dict]:
    if not registry.is_file():
        return []
    out: list[dict] = []
    for line in registry.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def list_all(registry: Path = REGISTRY_FILE) -> list[dict]:
    """Devuelve todas las entradas del registro (orden de inserción)."""
    return _load(registry)


def find(run_id: str, registry: Path = REGISTRY_FILE) -> dict:
    """Busca un run por id. ``KeyError`` si no aparece."""
    for entry in _load(registry):
        if entry.get("run_id") == run_id:
            return entry
    raise KeyError(f"run_id no encontrado en {registry}: {run_id}")


def best(
    user: Optional[str] = None,
    by: str = "best_val_iou_fire",
    registry: Path = REGISTRY_FILE,
) -> Optional[dict]:
    """Mejor entrada según `by` para `user` (o todos si user=None)."""
    candidates = [
        e for e in _load(registry)
        if e.get(by) is not None
        and e.get("status") == "completed"
        and (user is None or e.get("user") == user)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e[by])


def run(run_id: str, registry: Path = REGISTRY_FILE) -> int:
    """Reproduce un run usando el config y el ckpt guardados (si existe)."""
    entry = find(run_id, registry)
    config = entry.get("config_path")
    ckpt = entry.get("best_ckpt_path")
    commit = (entry.get("git_commit") or "").split("[")[0]
    if commit and _have_git() and _repo_dirty() is not None and not _repo_dirty():
        # Sólo cambiamos de rama si el usuario no tiene cambios sin commitear.
        _try_checkout(commit)
    if not config:
        raise FileNotFoundError(
            f"el run {run_id} no tiene config_path registrado"
        )
    config_p = Path(config)
    if not config_p.is_file():
        raise FileNotFoundError(
            f"config del run {run_id} no existe: {config}"
        )
    cmd: List[str] = ["python", "train.py", "fit", "--config", str(config_p)]
    if ckpt and Path(ckpt).is_file():
        cmd += ["--ckpt_path", str(ckpt)]
    print("[reproduce]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)


def _have_git() -> bool:
    import shutil
    return shutil.which("git") is not None


def _repo_dirty() -> Optional[bool]:
    """True si hay cambios sin commitear, None si git no responde."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _try_checkout(commit: str) -> None:
    try:
        subprocess.check_call(
            ["git", "checkout", commit, "--", "."],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pass


__all__ = ["list_all", "find", "best", "run", "REGISTRY_FILE"]
