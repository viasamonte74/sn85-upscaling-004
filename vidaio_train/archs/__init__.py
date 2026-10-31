"""Model zoo: BasicVSR++ (primary) and RRDBNet / Real-ESRNet (single-image fallback)."""

from __future__ import annotations

from .basicvsrpp import BasicVSRPlusPlus
from .rrdbnet import RRDBNet
from .spynet import SpyNet, flow_warp

__all__ = ["BasicVSRPlusPlus", "RRDBNet", "SpyNet", "flow_warp", "build_model"]


def build_model(cfg: dict):
    """``cfg = {"type": "basicvsrpp", ...kwargs}``; see configs/*.yaml."""
    kind = cfg["type"].lower()
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    if kind in ("basicvsrpp", "basicvsr++", "bvspp"):
        return BasicVSRPlusPlus(**kwargs)
    if kind in ("rrdbnet", "realesrnet", "rrdb"):
        return RRDBNet(**kwargs)
    raise ValueError(f"unknown model type {cfg['type']!r}")
