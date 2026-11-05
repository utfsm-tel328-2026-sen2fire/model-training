from __future__ import annotations

from typing import Iterable, Literal

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler


def build_optimizer(
    name: Literal["adam", "adamw", "sgd"],
    params: Iterable,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> Adam | AdamW | SGD:
    """Construye el optimizador según el nombre. Replica ``Baseline_code/Train.py:123``.

    El paper usa ``optim.Adam`` con lr=1e-4, weight_decay=5e-4. AdamW también
    está disponible porque muchos setups lo prefieren por el desacoplamiento
    del WD.
    """
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"optimizador desconocido: {name!r}")


def build_scheduler(
    name: Literal["poly", "cosine", "none"],
    optimizer,
    max_steps: int,
    power: float = 0.9,
) -> LRScheduler | None:
    """Schedule de LR.

    - ``poly`` replica ``adjust_learning_rate`` de los autores: decae en ``step``
      siguiendo ``(1 - step/max_steps) ** power``. Default power=0.9 (paper).
    - ``cosine`` usa ``CosineAnnealingLR`` a lo largo de ``max_steps``.
    - ``none`` devuelve None — el módulo lo trata como "sin scheduler".
    """
    if name == "none":
        return None
    if name == "poly":
        return LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1.0 - step / max(1, max_steps)) ** power,
        )
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max_steps)
    raise ValueError(f"lr_schedule desconocido: {name!r}")
