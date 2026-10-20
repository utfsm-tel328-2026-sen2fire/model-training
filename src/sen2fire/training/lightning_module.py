from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from sen2fire.models import UNet
from .losses import CrossEntropyFireWeighted
from .optim import build_optimizer, build_scheduler


class Sen2FireSegModule(pl.LightningModule):
    """LightningModule para segmentación binaria fire / non-fire.

    Sigue el recipe del paper:

    - Optimizer: ``Adam`` (no AdamW).
    - Loss: ``CrossEntropyLoss(weight=[1, weight_fire])`` para el desbalance.
    - LR schedule: decaimiento polinomial ``(1 - step/max_steps) ** 0.9``
      aplicado por iteración (no por epoch).
    - Métricas: mIoU por clase, F1 por clase, precision/recall por clase y
      pixel-OA. La métrica principal es ``val/iou_fire`` (clase 1).

    Lightning se encarga del resto: ``save_hyperparameters`` congela el YAML en
    cada checkpoint para reproducir, ``ModelCheckpoint`` monitorea
    ``val/iou_fire`` y guarda sólo el mejor, ``Trainer`` aplica
    ``model.eval()``/``torch.no_grad()`` por hook en validación.
    """

    def __init__(
        self,
        in_channels: int = 12,
        n_classes: int = 2,
        base: int = 64,
        depth: int = 5,
        bilinear: bool = True,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        ce_weight_fire: float = 10.0,
        lr_schedule: str = "poly",
        max_steps: int = 5000,
        power: float = 0.9,
        interpolate_to: Optional[Tuple[int, int]] = (512, 512),
        sgd_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            base=base,
            depth=depth,
            bilinear=bilinear,
        )
        self.loss_fn = CrossEntropyFireWeighted(weight_fire=ce_weight_fire, n_classes=n_classes)

        # Métricas torchmetrics. Las instancias son atributos del módulo, así
        # Lightning las resetea por época.
        self.val_iou = MulticlassJaccardIndex(num_classes=n_classes, average=None)
        self.val_f1 = MulticlassF1Score(num_classes=n_classes, average=None)
        self.val_precision = MulticlassPrecision(num_classes=n_classes, average=None)
        self.val_recall = MulticlassRecall(num_classes=n_classes, average=None)
        self.val_oa = MulticlassAccuracy(num_classes=n_classes, average="micro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        size = self.hparams.interpolate_to
        if size is not None:
            h, w = size
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        if logits.shape[-2:] != y.shape[-2:]:
            # Compatibilidad: y viene en (B, H, W) y logits puede tener otra
            # resolución si ``interpolate_to`` está apagado y el patch no es
            # uniforme. Re-sampleamos el target.
            y = F.interpolate(
                y.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()
        loss = self.loss_fn(logits, y)

        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss, logits.detach(), y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch, "val")
        preds = logits.argmax(dim=1)
        iou = self.val_iou(preds, y)
        f1 = self.val_f1(preds, y)
        prec = self.val_precision(preds, y)
        rec = self.val_recall(preds, y)
        oa = self.val_oa(preds, y)

        for idx, name in enumerate(("non_fire", "fire")):
            self.log(f"val/iou_{name}", iou[idx], on_epoch=True, prog_bar=(name == "fire"))
            self.log(f"val/f1_{name}", f1[idx], on_epoch=True, prog_bar=False)
            self.log(f"val/precision_{name}", prec[idx], on_epoch=True, prog_bar=False)
            self.log(f"val/recall_{name}", rec[idx], on_epoch=True, prog_bar=False)

        # Métrica secundaria: pixel-OA (accuracy micro).
        self.log("val/oa", oa, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.hparams.optimizer,
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=self.hparams.sgd_momentum,
        )
        scheduler = build_scheduler(
            self.hparams.lr_schedule,
            optimizer,
            max_steps=self.hparams.max_steps,
            power=self.hparams.power,
        )
        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @rank_zero_only
    def on_train_start(self) -> None:
        n = sum(p.numel() for p in self.parameters())
        msg = f"UNet params={n/1e6:.2f}M base={self.hparams.base} depth={self.hparams.depth} bilinear={self.hparams.bilinear}"
        self.log("model/info", 1.0, prog_bar=False, logger=False, batch_size=1)
        if self.logger is not None and hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.add_text("model/summary", msg)
            except Exception:
                pass
