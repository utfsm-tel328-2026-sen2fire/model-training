"""Entry point de entrenamiento y validación/testeo vía PyTorch Lightning CLI.

Uso:

    # Entrenar
    python train.py fit --config configs/baseline.yaml

    # Reentrenar / retomar desde un checkpoint
    python train.py fit --config configs/baseline.yaml --ckpt_path runs/<user>/<run>/checkpoints/best.ckpt

    # Validar el mejor checkpoint actual
    python train.py validate --config configs/baseline.yaml --ckpt_path runs/.../checkpoints/best.ckpt

    # Testear el mejor checkpoint
    python train.py test --config configs/baseline.yaml --ckpt_path runs/.../checkpoints/best.ckpt

Cualquier flag del YAML se puede sobreescribir por CLI:

    python train.py fit --config configs/baseline.yaml trainer.max_steps=2000 model.base=32

Para reproducir un run pasado desde la línea de `experiments.jsonl`:

    python -c "from sen2fire.training.reproduce import run; run('<run_id>')"

Después del fit (o de un test), el YAML usado se copia a
``<runs_root>/<user>/<run_id>/configs/config.yaml`` para que la línea en
``experiments.jsonl`` apunte a un archivo concreto (no a una ruta abstracta).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from pytorch_lightning.cli import LightningCLI

from sen2fire.data import SegmentationDataModule
from sen2fire.training.lightning_module import Sen2FireSegModule


def _parse_config_arg(argv: list[str]) -> Path | None:
    """Extrae el valor de ``--config <path>`` del ``argv``.

    Soporta tanto ``--config <path>`` como ``--config=<path>``.
    """
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in {"--config", "-c"} and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--config="):
            return Path(a.split("=", 1)[1])
        i += 1
    return None


def main() -> None:
    cli = LightningCLI(
        model_class=Sen2FireSegModule,
        datamodule_class=SegmentationDataModule,
        run=True,
        save_config_callback=None,
        seed_everything_default=True,
        parser_kwargs={"parser_mode": "yaml"},
    )

    # Snapshot del YAML bajo el árbol del run.
    config_arg = _parse_config_arg(sys.argv)
    trainer = cli.trainer
    if (
        config_arg is not None
        and trainer.is_global_zero
        and trainer.log_dir is not None
    ):
        try:
            log_root = Path(trainer.log_dir)
            run_root = log_root.parent if log_root.name == "tb_logs" else log_root
            snap = run_root / "configs" / "config.yaml"
            snap.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_arg, snap)
            print(f"[train.py] snapshot del config en {snap}")
        except OSError as e:
            print(f"[train.py] no se pudo hacer snapshot del config: {e}")


if __name__ == "__main__":
    main()
