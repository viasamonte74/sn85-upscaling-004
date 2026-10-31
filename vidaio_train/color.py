"""Colour conversions that reproduce the validator's pixel space exactly.

The scorer canonicalizes every stream to 8-bit yuv420p and converts the planes to RGB with
``cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)``. OpenCV ignores the BT.709 tags and applies
the ITU-R BT.601 limited-range matrix with nearest-neighbour chroma upsampling. The model
therefore works in *that* RGB space, and on output we build the 4:2:0 planes ourselves with
the exact inverse and piped as raw planes (so ffmpeg performs no colour conversion at all).

The *model input* is the one place where this convention is deliberately not followed: a
4:2:0 frame decoded with nearest chroma hands the network hard 2x2 chroma blocks (8x8 at the
x4 output scale), which it amplifies into luma noise. Measured with the stock BasicVSR++
weights on a practice clip (PSNR over the validator's Lanczos floor): nearest -2.65 dB,
bilinear -0.33 dB, cubic +0.68 dB. So ``chroma="cubic"`` (``MODEL_INPUT_CHROMA``) is used for
LR frames fed to the model, and the exact nearest decode for targets, floors and outputs.

Two implementations are provided:

* NumPy (uint8, exact integer behaviour) for data preparation and inference;
* torch (float, differentiable) for the chroma-subsampling round trip inside the loss.
"""

from __future__ import annotations

import numpy as np

# BT.601 limited range, the matrix OpenCV uses for COLOR_YUV2RGB_I420 (CV_YUV2RGB coefficients
# ITUR_BT_601_CY = 1.164383, CVR = 1.596027, CVG = -0.812968, CUG = -0.391762, CUB = 2.017232).
_CY, _CVR, _CVG, _CUG, _CUB = 1.164383, 1.596027, -0.812968, -0.391762, 2.017232
# RGB -> YCbCr (BT.601, limited range), the inverse of the above.
_KR, _KG, _KB = 0.299, 0.587, 0.114


#: chroma upsampling used for frames fed to the model (targets/floors/outputs always use "nearest")
MODEL_INPUT_CHROMA = "cubic"


def i420_to_rgb(i420: np.ndarray, width: int, height: int, chroma: str = "nearest") -> np.ndarray:
    """Planes (h*3/2, w) uint8 -> RGB (h, w, 3) uint8 with the scorer's BT.601 limited matrix.

    ``chroma="nearest"`` is exactly the scorer (OpenCV ``COLOR_YUV2RGB_I420``, bit-exact when
    OpenCV is present). ``"linear"`` / ``"cubic"`` upsample the chroma planes smoothly first,
    for the model input only (see the module docstring)."""
    if chroma == "nearest":
        try:
            import cv2

            return cv2.cvtColor(i420.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_I420)
        except ImportError:  # pragma: no cover - the training rig has OpenCV
            pass
    plane = width * height
    flat = i420.reshape(-1)
    y = flat[:plane].reshape(height, width).astype(np.float32)
    u = flat[plane : plane + plane // 4].reshape(height // 2, width // 2).astype(np.float32)
    v = flat[plane + plane // 4 :].reshape(height // 2, width // 2).astype(np.float32)
    if chroma == "nearest":
        u = u.repeat(2, axis=0).repeat(2, axis=1)
        v = v.repeat(2, axis=0).repeat(2, axis=1)
    else:
        import cv2

        interp = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[chroma]
        u = cv2.resize(u, (width, height), interpolation=interp)
        v = cv2.resize(v, (width, height), interpolation=interp)
    yy = _CY * (y - 16.0)
    r = yy + _CVR * (v - 128.0)
    g = yy + _CVG * (v - 128.0) + _CUG * (u - 128.0)
    b = yy + _CUB * (u - 128.0)
    return np.clip(np.rint(np.stack([r, g, b], -1)), 0, 255).astype(np.uint8)


def rgb_to_i420(rgb: np.ndarray) -> np.ndarray:
    """RGB (h, w, 3) uint8 -> I420 planes (h*3/2, w) uint8 (BT.601 limited, 2x2 chroma mean).

    h and w must be even. This is the inverse of :func:`i420_to_rgb` up to rounding, so a
    frame produced by the model survives the encode -> scorer decode with no colour shift."""
    h, w = rgb.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f"I420 needs even dimensions, got {w}x{h}")
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 16.0 + (219.0 / 255.0) * (_KR * r + _KG * g + _KB * b)
    cb = 128.0 + (224.0 / 255.0) * ((b - (_KR * r + _KG * g + _KB * b)) / (2.0 * (1.0 - _KB)))
    cr = 128.0 + (224.0 / 255.0) * ((r - (_KR * r + _KG * g + _KB * b)) / (2.0 * (1.0 - _KR)))
    cb = cb.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    cr = cr.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    out = np.empty(h * w * 3 // 2, np.uint8)
    out[: h * w] = np.clip(np.rint(y), 16, 235).astype(np.uint8).reshape(-1)
    out[h * w : h * w + h * w // 4] = np.clip(np.rint(cb), 16, 240).astype(np.uint8).reshape(-1)
    out[h * w + h * w // 4 :] = np.clip(np.rint(cr), 16, 240).astype(np.uint8).reshape(-1)
    return out.reshape(h * 3 // 2, w)


def luma_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """8-bit luma plane as the scorer sees it (the Y plane of the I420 frame)."""
    return rgb_to_i420(rgb)[: rgb.shape[0]]


# --------------------------------------------------------------------------- torch

def torch_rgb_to_ycbcr(x):
    """(n, 3, h, w) RGB in [0, 1] -> (Y, Cb, Cr) in 8-bit limited-range units (float)."""
    r, g, b = x[:, 0:1] * 255.0, x[:, 1:2] * 255.0, x[:, 2:3] * 255.0
    lum = _KR * r + _KG * g + _KB * b
    y = 16.0 + (219.0 / 255.0) * lum
    cb = 128.0 + (224.0 / 255.0) * ((b - lum) / (2.0 * (1.0 - _KB)))
    cr = 128.0 + (224.0 / 255.0) * ((r - lum) / (2.0 * (1.0 - _KR)))
    return y, cb, cr


def torch_ycbcr_to_rgb(y, cb, cr):
    """Inverse of :func:`torch_rgb_to_ycbcr` with OpenCV's coefficients; returns RGB in [0, 1]."""
    yy = _CY * (y - 16.0)
    r = yy + _CVR * (cr - 128.0)
    g = yy + _CVG * (cr - 128.0) + _CUG * (cb - 128.0)
    b = yy + _CUB * (cb - 128.0)
    import torch

    return torch.cat([r, g, b], dim=1).clamp_(0.0, 255.0) / 255.0


def torch_i420_roundtrip(x):
    """Differentiable 4:2:0 round trip: RGB -> YCbCr, 2x2 mean chroma, nearest upsample -> RGB.

    Applied to the model output before the pixel loss so the network is trained on what the
    scorer will measure after the encoder's chroma subsampling. No rounding is simulated."""
    import torch.nn.functional as F

    y, cb, cr = torch_rgb_to_ycbcr(x)
    cb = F.interpolate(F.avg_pool2d(cb, 2), scale_factor=2, mode="nearest")
    cr = F.interpolate(F.avg_pool2d(cr, 2), scale_factor=2, mode="nearest")
    return torch_ycbcr_to_rgb(y, cb, cr)


def torch_luma(x):
    """(n, 3, h, w) RGB in [0, 1] -> 8-bit luma (n, 1, h, w) float, the SSIM input."""
    return torch_rgb_to_ycbcr(x)[0]
