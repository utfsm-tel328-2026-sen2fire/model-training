"""Loop de entrenamiento mínimo.

Sirve como ejemplo canónico de cómo se usa `load_dataset()` en un loop real.
El modelo y la loss son intercambiables; el flujo del dataset es lo importante.

Uso:
    python train.py --epochs 1 --batch-size 2
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from sen2fire.data import SegmentationDataModule
from sen2fire.models import UNet, combined_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento mínimo Sen2Fire")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--dataset", default=os.environ.get("DATASET", "sen2fire"),
                   help="Nombre del dataset en el registro (ver sen2fire.data.registry)")
    p.add_argument("--env-file", default=".env",
                   help="Ruta al .env con las variables del dataset")
    p.add_argument("--limit-train", type=int, default=None,
                   help="Subset los primeros N patches de train (smoke test)")
    p.add_argument("--limit-val", type=int, default=None)
    return p.parse_args()


def run() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---------------------------------------------------------------
    # 1) LightningDataModule: encapsula fuente, dataset, splits y dataloaders.
    #    `prepare_data()` es idempotente y materializa la caché local una
    #    sola vez si está configurada; `setup("fit")` instancia los Subsets
    #    una única vez por instancia, así no relista la NFS en cada stage.
    # ---------------------------------------------------------------
    dm = SegmentationDataModule(
        name=args.dataset,
        env_file=args.env_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        limit_train=args.limit_train,
        limit_val=args.limit_val,
    )
    dm.prepare_data()
    dm.setup("fit")

    print(f"dataset:   {type(dm.ds).__name__}")
    print(f"root:     {dm.ds.root}")
    print(f"patches:  {len(dm.ds)}")
    print(f"split:    train={len(dm.train_dataset)}  val={len(dm.val_dataset)}")

    # ---------------------------------------------------------------
    # 2) DataLoaders del módulo. ds[i] devuelve (image float32 (12, 512, 512)
    #    normalizada, label long (512, 512) en {0, 1}).
    # ---------------------------------------------------------------
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # ---------------------------------------------------------------
    # 3) Modelo, loss y optimizador.
    # ---------------------------------------------------------------
    model = UNet(in_channels=12, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"modelo:   UNet ({n_params/1e6:.2f}M params)")

    # ---------------------------------------------------------------
    # 4) Loop de entrenamiento.
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
            loss = combined_loss(logits, label)

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
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                logits = model(image)
                loss = combined_loss(logits, label)

                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)

        dt = time.perf_counter() - t0
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({dt:.1f} s)")

    # Checkpoint al final
    ckpt_path = run_dir / "last.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"checkpoint guardado en {ckpt_path}")


if __name__ == "__main__":
    run()
