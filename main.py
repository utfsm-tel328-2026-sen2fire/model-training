from __future__ import annotations

import time

from torch.utils.data import DataLoader, Subset

from sen2fire.data import available_datasets, load_dataset
from sen2fire.data.datasets.sen2fire import get_scene_split


def main() -> None:
    t0 = time.perf_counter()
    ds = load_dataset()
    t_load = time.perf_counter() - t0

    print(f"dataset:   {type(ds).__name__}  (registrados: {', '.join(available_datasets())})")
    print(f"root:      {ds.root}")
    print(f"patches:   {len(ds)}")
    print(f"carga ds:  {t_load:.2f} s")

    split = get_scene_split(ds)
    print(f"split:     train={len(split['train'])}  val={len(split['val'])}  test={len(split['test'])}")

    img, lbl = ds[0]
    print(f"sample 0:  image={tuple(img.shape)} {img.dtype}  label={tuple(lbl.shape)} {lbl.dtype}")
    print(f"           image range=[{img.min().item():.4f}, {img.max().item():.4f}]  "
          f"label unique={lbl.unique().tolist()}")

    loader = DataLoader(
        Subset(ds, split["train"][:8]),
        batch_size=4,
        num_workers=2,
    )
    t0 = time.perf_counter()
    for batch_img, batch_lbl in loader:
        pass
    print(f"1 batch (4×512×512, num_workers=2): {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
