"""Crea un dataset Sen2Fire sintético en un directorio temporal.

Sirve para ejercitar el pipeline completo (caché, split, collate, loop) sin
depender de la montura NFS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_fake_scene(path: Path, n: int, size: int, seed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        image = rng.integers(0, 10000, size=(12, size, size), dtype=np.int16)
        label = (rng.random((size, size)) > 0.9).astype(np.uint8)
        aerosol = rng.random((size, size)).astype(np.float32)
        np.savez_compressed(path / f"patch_{i:04d}.npz",
                            image=image, aerosol=aerosol, label=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--per-scene", type=int, default=6)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    for s in range(1, args.scenes + 1):
        make_fake_scene(args.out / f"scene{s}", args.per_scene, args.size, seed=s)

    total = args.scenes * args.per_scene
    print(f"{total} patches en {args.out} "
          f"({args.scenes} escenas x {args.per_scene}, {args.size}x{args.size})")


if __name__ == "__main__":
    main()