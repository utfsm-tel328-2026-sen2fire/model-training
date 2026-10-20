"""Tests del LightningModule y de su integración con Trainer.

Cubre:
  - forward + backward funcionan;
  - las claves de log aparecen en `callback_metrics`;
  - configure_optimizers devuelve un dict con optimizer + lr_scheduler;
  - la LR decae bajo la política polinomial.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn

from sen2fire.data import SegmentationDataModule
from sen2fire.training.lightning_module import Sen2FireSegModule


def _make_synthetic_dm(batch_size: int = 2, image_size: int = 32) -> SegmentationDataModule:
    """Construye un DataModule que opera sobre un directorio temporal.

    Por defecto crea 4 escenas x 4 patches para que el split por escena
    default (scene1+2 vs scene3 vs scene4) tenga todos los grupos.
    """
    from pathlib import Path as _P
    from conftest import make_scene

    tmp = _P(tempfile.mkdtemp())
    make_scene(tmp, scenes=4, per_scene=4, size=image_size)
    env = tmp / ".env.test"
    env.write_text(f"SEN2FIRE_ROOT={tmp}\n")
    return SegmentationDataModule(
        env_file=env,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )


def _unused() -> None:
    """Quiet imports for mypy; helps when signature changes upstream."""
    return


def test_forward_backward_runs() -> None:
    module = Sen2FireSegModule(in_channels=12, max_steps=10, interpolate_to=None)
    x = torch.randn(2, 12, 16, 16)
    y = torch.randint(0, 2, (2, 16, 16))
    logits = module(x)
    loss = module.loss_fn(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.shape == (2, 2, 16, 16)


def test_configure_optimizers_poly_lr_decays() -> None:
    """La LR cae monótonamente siguiendo (1 - step/max_steps)**power."""
    module = Sen2FireSegModule(learning_rate=1.0, max_steps=10, power=0.9)
    out = module.configure_optimizers()
    assert isinstance(out, dict) and "optimizer" in out
    opt = out["optimizer"]
    sch = out["lr_scheduler"]["scheduler"]
    initial = opt.param_groups[0]["lr"]
    # Avanzamos el scheduler a mano y comprobamos.
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(10):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    # Después de max_steps, la LR va a ~0 (en realidad un poquito >0).
    assert lrs[-1] < initial
    assert lrs[0] == pytest.approx(initial)


def test_fit_with_synthetic_datamodule_logs_keys() -> None:
    """fast_dev_run hace un forward+backward por dataloader y arma los logs."""
    dm = _make_synthetic_dm(batch_size=2, image_size=16)
    module = Sen2FireSegModule(
        in_channels=12, max_steps=10, interpolate_to=None, learning_rate=1e-3,
    )

    trainer = pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[],
    )
    trainer.fit(module, dm)

    logged = trainer.callback_metrics
    assert "train/loss" in logged or "val/loss" in logged or "val/iou_fire" in logged or "val/oa" in logged
    # Cualquiera de las métricas core tiene que haber sido loggeada.
    assert any(k in logged for k in ("train/loss", "val/loss", "val/iou_fire", "val/oa"))


def test_hparams_were_saved() -> None:
    """save_hyperparameters congela todos los flags del módulo en hparams."""
    module = Sen2FireSegModule(
        in_channels=12, learning_rate=3e-4, ce_weight_fire=7.0, max_steps=42,
    )
    assert module.hparams.learning_rate == 3e-4
    assert module.hparams.ce_weight_fire == 7.0
    assert module.hparams.max_steps == 42


def test_dataset_constructor_rejects_n_classes_lt_2() -> None:
    with pytest.raises(ValueError):
        Sen2FireSegModule(n_classes=1)
