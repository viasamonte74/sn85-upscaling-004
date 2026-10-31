"""RRDBNet (ESRGAN / Real-ESRNet backbone) — the single-image fallback and ablation baseline.

Names mirror BasicSR's ``rrdbnet_arch.py`` so ``RealESRNet_x4plus.pth`` (BSD-3-Clause, the
L1-trained model that already holds the best public score on the practice pack) loads
directly. x2 is handled the Real-ESRGAN way: pixel-unshuffle the input to 12 channels and
reuse the x4 head (``num_in_ch=12``), exactly as ``RealESRGAN_x2plus`` does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basicvsrpp import default_init_weights


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRNet x4plus: num_in_ch=3, num_feat=64, num_block=23, num_grow_ch=32 (16.7 M params).

    ``forward(x, scale)`` accepts (n, 3, h, w) or (n, t, 3, h, w); scale 2 pixel-unshuffles."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64, num_block: int = 23,
                 num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_first_x2 = nn.Conv2d(12, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def init_x2_stem_from_x4(self) -> None:
        with torch.no_grad():
            self.conv_first_x2.weight.copy_(self.conv_first.weight.repeat(1, 4, 1, 1) / 4.0)
            self.conv_first_x2.bias.copy_(self.conv_first.bias)

    def forward(self, x: torch.Tensor, scale: int = 4) -> torch.Tensor:
        video = x.dim() == 5
        if video:
            n, t = x.shape[:2]
            x = x.reshape(-1, *x.shape[2:])
        if scale == 2:
            feat = self.conv_first_x2(F.pixel_unshuffle(x, 2))
        elif scale == 4:
            feat = self.conv_first(x)
        else:
            raise ValueError("scale must be 2 or 4")
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        if video:
            out = out.view(n, t, *out.shape[1:])
        return out
