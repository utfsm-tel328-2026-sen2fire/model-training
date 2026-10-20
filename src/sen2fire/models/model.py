from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Up block del U-Net de los autores de Sen2Fire (`Baseline_code/model/Networks.py`).

    En modo bilinear usa ``Upsample(scale_factor=2, mode='bilinear', align_corners=True)``
    que preserva los canales; en modo ConvT usa ``ConvTranspose2d`` que los divide a la
    mitad. En ambos casos, ``in_channels`` representa los canales **post-cat**
    (skip + up), no los canales del tensor bajo de resolución que entra al bloque.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net con la profundidad y el head del paper de Sen2Fire.

    Replica exacto a ``Baseline_code/model/Networks.py:62-91``. Defaults:

        in_channels=12, n_classes=2, base=64, depth=5, bilinear=True

    Args:
        in_channels: canales de entrada. 12 para todas las bandas Sentinel-2 sin
            aerosol (modo 0 de los autores); 13 si se incluye la banda de
            aerosol (modo 1).
        n_classes: número de clases para softmax. 2 para el setup binario
            fire/non-fire de los autores; cualquier K >= 2 funciona.
        base: número de canales en la primera etapa de la U (canales del
            ``inc``). Los autores usan 64.
        depth: profundidad del U-Net. Por ahora sólo 5 (la del paper).
        bilinear: si True, ``Upsample`` bilinear; si False, ``ConvTranspose2d``.

    El input debe tener un tamaño espacial divisible por 2^(depth-1) = 16
    cuando depth=5 (los autores usan parches de 512x512).
    """

    def __init__(
        self,
        in_channels: int = 12,
        n_classes: int = 2,
        base: int = 64,
        depth: int = 5,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes debe ser >= 2 (recibido {n_classes})")
        if depth != 5:
            raise ValueError(f"depth debe ser 5 (recibido {depth}); usar Networks.py del paper")

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, (base * 16) // factor)
        self.up1 = Up(base * 16, (base * 8) // factor, bilinear)
        self.up2 = Up(base * 8, (base * 4) // factor, bilinear)
        self.up3 = Up(base * 4, (base * 2) // factor, bilinear)
        self.up4 = Up(base * 2, base, bilinear)
        self.outc = OutConv(base, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.down4(s4)
        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.outc(x)
