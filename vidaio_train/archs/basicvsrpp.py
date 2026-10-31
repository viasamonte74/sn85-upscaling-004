"""BasicVSR++ (Chan et al., CVPR 2022) in plain PyTorch, with a shared-cost x2 path.

Why this architecture: see README "Model". Design points specific to this competition:

* **Names mirror BasicSR's ``basicvsrpp_arch.py``** so the Apache-2.0 weights published by
  OpenMMLab / BasicSR load directly (``convert_weights.py`` handles mmagic's prefixes).
* **Deformable alignment uses ``torchvision.ops.deform_conv2d``** (modulated, with mask), so
  the contender image needs no mmcv and no CUDA build; torch import time (which the speed
  term counts on every batch) stays small.
* **x2 items are run through pixel-unshuffle** (960x540x3 -> 480x270x12) into the same
  270p backbone with the x4 reconstruction head, so a 1080p output frame costs the same
  whether the item is x2 or x4. The x2 stem (``feat_extract_x2``) is the only extra module.
* No ``cpu_cache`` path: inference chunks long clips instead (``vidaio_train.infer``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _deform_conv2d_op():
    """torchvision's modulated deformable conv WITHOUT importing the torchvision package (its
    import costs ~1.2 s per run.sh call, i.e. per scored batch; only the C++ op is needed).
    Falls back to the regular import if the extension cannot be located."""
    import torch

    try:
        import glob
        import importlib.util
        import os

        spec = importlib.util.find_spec("torchvision")
        lib = sorted(glob.glob(os.path.join(spec.submodule_search_locations[0], "_C*.so")))[0]
        torch.ops.load_library(lib)
        torch.ops.torchvision.deform_conv2d  # noqa: B018 - raises if the op is missing
    except Exception:  # noqa: BLE001 - any failure: plain import
        import torchvision  # noqa: F401
    return torch.ops.torchvision.deform_conv2d


_DEFORM_CONV2D = _deform_conv2d_op()


def deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask):
    """Same contract as ``torchvision.ops.deform_conv2d`` (modulated, with mask) on the raw op."""
    kh, kw = weight.shape[-2:]
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    groups = x.shape[1] // weight.shape[1]
    offset_groups = offset.shape[1] // (2 * kh * kw)
    return _DEFORM_CONV2D(x, weight, offset, mask, bias, sh, sw, ph, pw, dh, dw, groups, offset_groups, True)

from .spynet import SpyNet, flow_warp


def default_init_weights(modules, scale: float = 1.0) -> None:
    for m in modules:
        for sub in m.modules():
            if isinstance(sub, nn.Conv2d):
                nn.init.kaiming_normal_(sub.weight, a=0, mode="fan_in")
                sub.weight.data *= scale
                if sub.bias is not None:
                    sub.bias.data.zero_()


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat: int = 64, res_scale: float = 1.0) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x))) * self.res_scale


class ConvResidualBlocks(nn.Module):
    """Conv + LeakyReLU + ``num_block`` residual blocks (BasicSR naming: ``main.0``, ``main.2.i``)."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 64, num_block: int = 15) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(num_feat=num_out_ch) for _ in range(num_block)]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class SecondOrderDeformableAlignment(nn.Module):
    """Flow-guided second-order deformable alignment (Eq. 6 of the paper), modulated DCN v2.

    Parameters ``weight``/``bias`` are the DCN kernel (same names as BasicSR's
    ``ModulatedDeformConvPack``); ``conv_offset`` predicts offsets and masks from the
    warped neighbours, the current feature and the two flows."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1,
                 deformable_groups: int = 16, max_residue_magnitude: float = 10.0) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.padding = kernel_size, padding
        self.stride, self.dilation = 1, 1
        self.deformable_groups = deformable_groups
        self.max_residue_magnitude = max_residue_magnitude
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=1)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels + 4, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deformable_groups, 3, 1, 1),
        )
        nn.init.constant_(self.conv_offset[-1].weight, 0.0)
        nn.init.constant_(self.conv_offset[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, extra_feat: torch.Tensor, flow_1: torch.Tensor,
                flow_2: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        mask = torch.sigmoid(mask)
        # torchvision's deform_conv2d does not support bf16/fp16 on every build: run it in fp32.
        dt = x.dtype
        return deform_conv2d(
            x.float(), offset.float(), self.weight.float(), self.bias.float(),
            self.stride, self.padding, self.dilation, mask.float(),
        ).to(dt)


class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ for the faithful upscaling track.

    Args:
        mid_channels: feature width (64 for the c64n7 weights, 128 for c128n25).
        num_blocks: residual blocks per propagation branch (7 or 25).
        max_residue_magnitude: cap on the learned offset residue.
        x2_mode: ``"unshuffle"`` runs x2 inputs through pixel-unshuffle into the same backbone
            (recommended, uniform cost); ``"downscale"`` runs the x4 path on the area-downscaled
            input and discards nothing (lower fidelity, no extra parameters).

    forward(lqs, scale) with ``lqs`` (n, t, 3, h, w) in [0, 1] returns (n, t, 3, scale*h, scale*w).
    """

    def __init__(self, mid_channels: int = 64, num_blocks: int = 7, max_residue_magnitude: float = 10.0,
                 x2_mode: str = "unshuffle", spynet_trainable: bool = True) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.x2_mode = x2_mode
        self.spynet = SpyNet()
        self.spynet.requires_grad_(spynet_trainable)
        self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)
        # x2 stem: the pixel-unshuffled 12-channel frame shares the residual blocks of the x4 stem.
        self.feat_extract_x2 = nn.Conv2d(12, mid_channels, 3, 1, 1, bias=True)
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3, padding=1, deformable_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)
        self.reconstruction = ConvResidualBlocks(5 * mid_channels, mid_channels, 5)
        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # ------------------------------------------------------------------ pieces
    def init_x2_stem_from_x4(self) -> None:
        """Initialise the 12-channel x2 stem from the 3-channel x4 stem (each phase gets 1/4)."""
        with torch.no_grad():
            w = self.feat_extract.main[0].weight  # (mid, 3, 3, 3)
            self.feat_extract_x2.weight.copy_(w.repeat(1, 4, 1, 1) / 4.0)
            self.feat_extract_x2.bias.copy_(self.feat_extract.main[0].bias)

    def compute_flow(self, lqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def propagate(self, feats: dict, flows: torch.Tensor, module_name: str) -> dict:
        n, t, _, h, w = flows.size()
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][idx]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i]]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1]]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ["spatial", module_name]] + [feat_prop]
            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=1))
            feats[module_name].append(feat_prop)
        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def upsample(self, base: torch.Tensor, feats: dict) -> torch.Tensor:
        """``base`` (n, t, 3, H, W): the bilinear upscale of the served input (residual skip)."""
        outputs = []
        for i in range(base.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][i])
            hr = torch.cat(hr, dim=1)
            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            outputs.append(hr + base[:, i])
        return torch.stack(outputs, dim=1)

    # ------------------------------------------------------------------ forward
    def forward(self, lqs: torch.Tensor, scale: int = 4) -> torch.Tensor:
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        n, t, c, h, w = lqs.size()
        base = F.interpolate(lqs.reshape(-1, c, h, w), scale_factor=scale, mode="bilinear",
                             align_corners=False).view(n, t, c, scale * h, scale * w)
        if scale == 4:
            x = lqs
            flow_src = lqs
            spatial = self.feat_extract(x.reshape(-1, c, h, w))
        elif self.x2_mode == "unshuffle":
            if h % 2 or w % 2:
                raise ValueError("x2 inputs must have even dimensions")
            x = F.pixel_unshuffle(lqs.reshape(-1, c, h, w), 2)  # (n*t, 12, h/2, w/2)
            flow_src = F.interpolate(lqs.reshape(-1, c, h, w), scale_factor=0.5, mode="area").view(
                n, t, c, h // 2, w // 2)
            stem = self.lrelu(self.feat_extract_x2(x))
            spatial = self.feat_extract.main[2](stem)
        else:  # "downscale": run the x4 path on a half-resolution copy
            x = F.interpolate(lqs.reshape(-1, c, h, w), scale_factor=0.5, mode="area")
            flow_src = x.view(n, t, c, h // 2, w // 2)
            spatial = self.feat_extract(x)
        fh, fw = spatial.shape[-2:]
        spatial = spatial.view(n, t, -1, fh, fw)
        feats = {"spatial": [spatial[:, i] for i in range(t)]}
        assert fh >= 32 and fw >= 32, f"feature grid must be at least 32x32 (SpyNet has 5 pooling levels), got {fw}x{fh}"
        flows_forward, flows_backward = self.compute_flow(flow_src)
        for iter_ in (1, 2):
            for direction in ("backward", "forward"):
                module = f"{direction}_{iter_}"
                feats[module] = []
                flows = flows_backward if direction == "backward" else flows_forward
                feats = self.propagate(feats, flows, module)
        return self.upsample(base, feats)
