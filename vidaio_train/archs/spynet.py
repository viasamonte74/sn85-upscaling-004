"""SpyNet optical flow (the flow estimator BasicVSR++ uses) and flow warping.

Module names mirror BasicSR's ``spynet_arch.py`` so the published weights load directly
(``basic_module.{level}.basic_module.{2j}``). mmagic's checkpoints are remapped by
``vidaio_train.convert_weights``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(x: torch.Tensor, flow: torch.Tensor, interp_mode: str = "bilinear",
              padding_mode: str = "zeros", align_corners: bool = True) -> torch.Tensor:
    """Warp ``x`` (n, c, h, w) by ``flow`` (n, h, w, 2) given in pixels (x, y)."""
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=x.device, dtype=x.dtype),
                                    torch.arange(0, w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3).to(x.dtype)
    return F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)


class BasicModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(16, 2, 7, 1, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basic_module(x)


class SpyNet(nn.Module):
    """Six-level SpyNet. Inputs are RGB in [0, 1]; returns flow (n, 2, h, w) in pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def process(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]
        for _ in range(5):
            ref.insert(0, F.avg_pool2d(ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(supp[0], kernel_size=2, stride=2, count_include_pad=False))
        flow = ref[0].new_zeros([ref[0].size(0), 2, int(math.floor(ref[0].size(2) / 2.0)),
                                 int(math.floor(ref[0].size(3) / 2.0))])
        for level in range(len(ref)):
            up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            if up.size(2) != ref[level].size(2):
                up = F.pad(up, pad=[0, 0, 0, 1], mode="replicate")
            if up.size(3) != ref[level].size(3):
                up = F.pad(up, pad=[0, 1, 0, 0], mode="replicate")
            warped = flow_warp(supp[level], up.permute(0, 2, 3, 1), interp_mode="bilinear", padding_mode="border")
            flow = self.basic_module[level](torch.cat([ref[level], warped, up], 1)) + up
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        assert ref.size() == supp.size()
        h, w = ref.size(2), ref.size(3)
        w_floor = int(math.ceil(w / 32.0) * 32)
        h_floor = int(math.ceil(h / 32.0) * 32)
        ref = F.interpolate(ref, size=(h_floor, w_floor), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_floor, w_floor), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.process(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        flow = torch.stack([flow[:, 0] * (float(w) / float(w_floor)), flow[:, 1] * (float(h) / float(h_floor))], 1)
        return flow
