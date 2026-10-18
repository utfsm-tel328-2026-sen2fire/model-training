"""Loop de entrenamiento mínimo.

Sirve como ejemplo canónico de cómo se usa `Sen2FireDataset` en un loop real.
El modelo y la loss son intercambiables; el flujo del dataset es lo importante.

Uso:
    python train.py --epochs 1 --batch-size 2
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from numpy._core.numeric import dtype

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset

from sen2fire.data import Sen2FireDataset, get_scene_split
from sen2fire.models import UNet

import csv

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento mínimo Sen2Fire")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--limit-train", type=int, default=None,
                   help="Subset los primeros N patches de train (smoke test)")
    p.add_argument("--limit-val", type=int, default=None)
    return p.parse_args()

def calculate_metrics(logits, labels, num_classes = 2):
    # Convierte los logits a predicciones concretas
    preds = torch.argmax(logits, dim=1)

    iou_sum = 0.0
    dice_sum = 0.0
    valid_classes = 0

    for cls in range(1, num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)

        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()

        if union > 0:
            iou_sum += intersection / union
            dice_sum += (2 * intersection) / (pred_mask.sum() + label_mask.sum())
            valid_classes += 1

    # Si no hay evita la division por 0 ;v
    if valid_classes == 0:
        return 0.0, 0.0

    return (iou_sum / valid_classes), (dice_sum / valid_classes)

def early_stopping(val_losses, patience=5):
    if len(val_losses) <= patience: # Su no hay suficientes perdidad no hay early stopping
        return False

    best_loss = min(val_losses[:-patience])
    recent_losses = val_losses[-patience:]
    # return True si todas las pérdidas recientes son mayores que la mejor pérdida anterior
    return all(loss >= best_loss for loss in recent_losses)


def run() -> None:

    enable_EarlyStopping = input("Activar Early Stoppign y/n: ")
    if enable_EarlyStopping.lower() == 'y':
        enable_EarlyStopping = True
    else:
        enable_EarlyStopping = False
    # Carpeta del historial de resultados de entrenamiento.
    record_result_path = Path("records")
    record_result_path.mkdir(exist_ok=True)

    # Archivo CSV para guardar los resultados del entrenamiento.
    fecha_hora = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    path_csv = record_result_path / f"training_results_{fecha_hora}.csv"
    headings = ["epoch", "train_loss", "val_loss", "val_IoU", "val_Dice", "time"]

    val_losses = []
    best_val_loss = float('inf')

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---------------------------------------------------------------
    # 1) Cargar el dataset.
    #    Sen2FireDataset() lee .env, clona el dataset al caché local
    #    si hace falta (configurado en SEN2FIRE_LOCAL_CACHE), e
    #    indexa todos los .npz encontrados.
    # ---------------------------------------------------------------
    ds = Sen2FireDataset()
    print(f"root:     {ds.root}")
    print(f"patches:  {len(ds)}")

    # ---------------------------------------------------------------
    # 2) Split por escena (train: scene1+scene2, val: scene3, test: scene4).
    #    Devuelve índices en el mismo orden que ds.samples.
    # ---------------------------------------------------------------
    split = get_scene_split(ds.samples)
    train_idx = split["train"]
    val_idx = split["val"]
    if args.limit_train is not None:
        train_idx = train_idx[: args.limit_train]
    if args.limit_val is not None:
        val_idx = val_idx[: args.limit_val]
    print(f"split:    train={len(train_idx)}  val={len(val_idx)}")

    # ---------------------------------------------------------------
    # 3) Subset + DataLoader — el patrón estándar de PyTorch.
    #    ds[i] devuelve (image, label) con image float32 (12, 512, 512)
    #    normalizada a reflectancia y label long (512, 512) en {0, 1}.
    # ---------------------------------------------------------------
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---------------------------------------------------------------
    # 4) Modelo, loss y optimizador.
    # ---------------------------------------------------------------
    model = UNet(in_channels=12, out_channels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    class_weights = torch.tensor([1.0, 10.0], dtype=torch.float32,device=device)  # Ajusta según tus necesidades
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"modelo:   UNet ({n_params/1e6:.2f}M params)")

    # ---------------------------------------------------------------
    # 5) Loop de entrenamiento.
    # ---------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for image, label in train_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits = model(image)
            loss = criterion(logits, label.squeeze(1).long())

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_sum += loss.item()
            train_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)

        # Validación: una pasada por val_loader, modo eval, sin grad.
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_iou_sum = 0.0
        val_dice_sum = 0.0

        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits = model(image)
                loss = criterion(logits, label.squeeze(1).long())

                batch_iou, batch_dice = calculate_metrics(logits, label)

                val_iou_sum += batch_iou
                val_dice_sum += batch_dice
                val_loss_sum += loss.item()
                val_batches += 1


        val_loss = val_loss_sum / max(val_batches, 1)
        val_iou = val_iou_sum / max(val_batches, 1)
        val_dice = val_dice_sum / max(val_batches, 1)
        val_losses.append(val_loss)
        dt = time.perf_counter() - t0

        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_IoU={val_iou:.4f}  val_Dice={val_dice:.4f}  "
              f"({dt:.1f} s)")

        # Guardar resultados en CSV
        with open(path_csv, mode="a", newline="", encoding="utf-8") as archive:
            writer = csv.writer(archive)
            if epoch == 0:
                writer.writerow(headings)
            writer.writerow([epoch + 1, train_loss, val_loss, val_iou, val_dice, dt])

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss},run_dir / "best.pt")

        if early_stopping(val_losses, patience=5) and enable_EarlyStopping:
            print(f"Early stopping activado en epoch {epoch + 1}. Mejor val_loss: {best_val_loss:.4f}")

            with open(path_csv, mode="a", newline="", encoding="utf-8") as archive:
                writer = csv.writer(archive)
                writer.writerow(["Early stopping activado en epoch", epoch + 1, "Mejor val_loss:", best_val_loss])

            break

    # Checkpoint al final
    ckpt_path = run_dir / "last.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"checkpoint guardado en {ckpt_path}")


if __name__ == "__main__":
    run()
