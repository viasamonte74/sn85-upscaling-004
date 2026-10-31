"""Contract-compliant inference: one degraded input + task JSON -> one 1080p MP4.

Memory: BasicVSR++ keeps five feature maps per frame; at 128 channels and 270p that is
~80 MB per frame in fp16, so a 240-frame clip is chunked (``chunk`` frames with ``overlap``
frames of context on each side, centre frames kept). With ``chunk >= frames`` the whole
clip is one bidirectional pass, which is best for quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from .archs import build_model
from .color import MODEL_INPUT_CHROMA, rgb_to_i420, torch_rgb_to_ycbcr, torch_ycbcr_to_rgb
from .ffio import ContractEncoder, iter_i420, iter_rgb, probe

SIDECAR_PREFIX = ".vidaio-next-upscale-task-"


def load_checkpoint(path: str, device: str, override: dict | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {"type": ckpt.get("arch", "basicvsrpp")}
    if cfg["type"] == "basicvsrpp":
        cfg.update({"mid_channels": ckpt.get("mid_channels", 64), "num_blocks": ckpt.get("num_blocks", 7)})
    elif cfg["type"] == "rrdbnet":
        cfg.update({"num_block": ckpt.get("num_blocks", 23)})
    if override:
        cfg.update(override)
    model = build_model(cfg)
    sd = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device)


def load_factor_models(path: str, device: str) -> dict:
    """{2: model, 4: model}. A checkpoint may carry separate ×4 weights (``params_ema_x4``, same
    architecture) when a different fine-tune is better for ×4 items; otherwise both factors share
    one model. The factor of every item is known from its task sidecar before it is processed."""
    model = load_checkpoint(path, device)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("params_ema_x4") is None:
        return {2: model, 4: model}
    model4 = load_checkpoint(path, device)
    model4.load_state_dict(ckpt["params_ema_x4"], strict=True)
    return {2: model, 4: model4.eval()}


def read_task(input_path: Path) -> dict:
    sidecar = input_path.parent / f"{SIDECAR_PREFIX}{input_path.name}"
    with open(sidecar) as fh:
        return json.load(fh)


@torch.no_grad()
def upscale_frames(model, frames: np.ndarray, scale: int, device: str, chunk: int = 64, overlap: int = 8,
                   dtype=torch.float16) -> np.ndarray:
    """``frames`` (t, h, w, 3) uint8 -> (t, scale*h, scale*w, 3) uint8."""
    t = frames.shape[0]
    out = np.empty((t, frames.shape[1] * scale, frames.shape[2] * scale, 3), np.uint8)
    starts = list(range(0, t, chunk)) if chunk < t else [0]
    for s in starts:
        e = min(t, s + chunk)
        cs, ce = max(0, s - overlap), min(t, e + overlap)
        x = torch.from_numpy(frames[cs:ce]).to(device).permute(0, 3, 1, 2).float().div_(255.0)[None]
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype,
                            enabled=device.startswith("cuda")):
            y = model(x, scale=scale)
        y = (y[0, s - cs : e - cs].float().clamp_(0, 1) * 255.0).round_().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        out[s:e] = y
    return out


def _i420_planes(i420: torch.Tensor, width: int, height: int):
    """(n, h*3/2, w) uint8/float -> Y (n,1,h,w), Cb (n,1,h/2,w/2), Cr (n,1,h/2,w/2) float."""
    n = i420.shape[0]
    flat = i420.reshape(n, -1).float()
    plane = width * height
    y = flat[:, :plane].reshape(n, 1, height, width)
    cb = flat[:, plane : plane + plane // 4].reshape(n, 1, height // 2, width // 2)
    cr = flat[:, plane + plane // 4 :].reshape(n, 1, height // 2, width // 2)
    return y, cb, cr


def torch_i420_to_rgb(i420: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """GPU twin of :func:`vidaio_train.color.i420_to_rgb` with ``chroma="cubic"`` (the model's input
    convention): bicubic chroma upsampling with half-pixel centres, OpenCV's BT.601 matrix.
    Returns (n, 3, h, w) float in [0, 1]."""
    import torch.nn.functional as F

    y, cb, cr = _i420_planes(i420, width, height)
    cb = F.interpolate(cb, size=(height, width), mode="bicubic", align_corners=False)
    cr = F.interpolate(cr, size=(height, width), mode="bicubic", align_corners=False)
    return torch_ycbcr_to_rgb(y, cb, cr)


def torch_rgb_to_i420(rgb: torch.Tensor) -> torch.Tensor:
    """GPU twin of :func:`vidaio_train.color.rgb_to_i420`: (n, 3, h, w) RGB in [0, 1] ->
    (n, h*3/2, w) uint8 I420 (BT.601 limited, 2x2 mean chroma, legal-range clamps)."""
    import torch.nn.functional as F

    n, _, h, w = rgb.shape
    y, cb, cr = torch_rgb_to_ycbcr(rgb)
    y = y.round_().clamp_(16, 235)
    cb = F.avg_pool2d(cb, 2).round_().clamp_(16, 240)
    cr = F.avg_pool2d(cr, 2).round_().clamp_(16, 240)
    return torch.cat([y.reshape(n, -1), cb.reshape(n, -1), cr.reshape(n, -1)], dim=1).to(torch.uint8).reshape(n, h * 3 // 2, w)


def _is_oom(exc: BaseException) -> bool:
    """CUDA OOM arrives as torch.cuda.OutOfMemoryError from PyTorch ops but as a plain RuntimeError
    when raised inside torchvision's deformable-conv kernel (across the libtorch C API boundary)."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc))


@torch.no_grad()
def _process_item_gpu(model, input_path: Path, output_path: Path, device: str, encoder: str, quality: int,
                      chunk: int, overlap: int, info, tw: int, th: int, factor: int, alpha: float,
                      dtype=torch.float16, prefetched: np.ndarray | None = None, defer_finish: bool = False):
    """Fast path: colour conversion and the floor blend on the GPU, x264 in a writer thread.

    The CPU path spent ~100 ms/frame in numpy RGB->I420, ~55 ms in cubic chroma upsampling and
    ~17 ms in the blend, serialised with the model and the encoder; here the GPU does those in
    ~1 ms and the encoder overlaps the next chunk's inference (speed is 30 % of the score).

    ``prefetched``: the input's I420 frames if the caller decoded them ahead of time (see
    :func:`read_input_i420`). ``defer_finish=True`` returns ``(stats, finish)`` as soon as the
    last chunk is queued, so the caller can start the next item's GPU work while this item's
    encoder drains; ``finish()`` joins the writer, closes the file and returns the final stats."""
    import queue
    import threading

    t0 = time.perf_counter()
    src = prefetched if prefetched is not None else read_input_i420(input_path, info)
    t = src.shape[0]
    t1 = time.perf_counter()
    floors: "queue.Queue | None" = None
    if alpha < 1.0:
        floors = queue.Queue(maxsize=max(16, chunk + overlap))

        def _read_floor():
            try:
                for fl in iter_i420(input_path, tw, th, vf=f"scale={tw}:{th}:flags=lanczos"):
                    floors.put(fl)
            finally:
                floors.put(None)

        threading.Thread(target=_read_floor, daemon=True).start()
    out_q: "queue.Queue" = queue.Queue(maxsize=max(16, chunk))
    enc_err: list[BaseException] = []
    enc = ContractEncoder(output_path, tw, th, Fraction(info.fps), encoder=encoder, quality=quality)

    def _write():
        try:
            while True:
                arr = out_q.get()
                if arr is None:
                    return
                for fr in arr:
                    enc.write_i420(fr)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            enc_err.append(exc)

    writer = threading.Thread(target=_write, daemon=True)
    writer.start()
    model_s = 0.0
    try:
        starts = list(range(0, t, chunk)) if chunk < t else [0]
        for s in starts:
            e = min(t, s + chunk)
            cs, ce = max(0, s - overlap), min(t, e + overlap)
            tm = time.perf_counter()
            x = torch_i420_to_rgb(torch.from_numpy(src[cs:ce]).to(device, non_blocking=True), info.width, info.height)[None]
            with torch.autocast(device_type="cuda", dtype=dtype):
                y = model(x, scale=factor)
            y = y[0, s - cs : e - cs].float().clamp_(0, 1)
            i420 = torch_rgb_to_i420(y)                                         # (k, H*3/2, W) uint8
            if floors is not None:
                fl = []
                for _ in range(e - s):
                    f = floors.get()
                    if f is None:
                        raise RuntimeError(f"{input_path.name}: floor decode ended early at frame {s + len(fl)}")
                    fl.append(f)
                flt = torch.from_numpy(np.stack(fl)).to(device, non_blocking=True).float()
                i420 = (flt + alpha * (i420.float() - flt)).round_().clamp_(0, 255).to(torch.uint8)
            frames_out = i420.cpu().numpy()
            model_s += time.perf_counter() - tm
            if enc_err:
                raise enc_err[0]
            out_q.put(frames_out)
        out_q.put(None)
    except BaseException:
        enc.close()
        raise
    t2 = time.perf_counter()

    def finish() -> dict:
        try:
            writer.join()
            if enc_err:
                raise enc_err[0]
        finally:
            enc.close()
        if enc.frames != t:
            raise RuntimeError(f"{input_path.name}: wrote {enc.frames} frames, input has {t}")
        t3 = time.perf_counter()
        return {"frames": int(t), "decode_s": t1 - t0, "model_s": model_s, "encode_s": (t3 - t1) - model_s,
                "drain_s": t3 - t2, "total_s": t3 - t0, "spf": (t3 - t0) / max(1, t), "path": "gpu"}

    if defer_finish:
        return {"frames": int(t), "decode_s": t1 - t0, "model_s": model_s, "gpu_done_s": t2 - t0}, finish
    return finish()


def read_input_i420(input_path: Path, info) -> np.ndarray:
    """Decode the whole (small) input to I420 frames, (t, h*3/2, w) uint8; safe to run in a thread
    ahead of the item so the GPU never waits for ffmpeg."""
    return np.stack(list(iter_i420(input_path, info.width, info.height)))


def process_item(model, input_path: Path, output_path: Path, device: str, encoder: str = "libx264",
                 quality: int = 10, chunk: int = 64, overlap: int = 8, task: dict | None = None,
                 floor_blend: dict[int, float] | None = None, prefetched: np.ndarray | None = None,
                 defer_finish: bool = False):
    """``floor_blend`` maps an upscale factor to alpha in (0, 1]: the written frame is
    ``floor + alpha * (model - floor)`` per I420 plane, where ``floor`` is the validator's own
    Lanczos upscale of the input (``scale=W:H:flags=lanczos`` on yuv420p). The fine-tuned model
    restores more high-frequency energy than grainy phone or smooth AI references hold, which
    drops PSNR below the floor; shrinking toward the floor fixes that (p01 0.156 -> 0.223,
    p10 0 -> 0.079 at alpha 0.6 with the real scorer) and improves LPIPS as well.

    On CUDA the GPU pipeline (:func:`_process_item_gpu`) is used; an out-of-memory error halves
    the chunk and falls back to the plain CPU-conversion path below (peak VRAM is ~21 GiB at
    chunk 96 for a 1080p output; the sandbox GPU is not ours to choose)."""
    task = task or read_task(input_path)
    tw, th, factor = int(task["target_width"]), int(task["target_height"]), int(task["upscale_factor"])
    info = probe(input_path)
    if (info.width * factor, info.height * factor) != (tw, th):
        raise ValueError(f"{input_path.name}: input {info.width}x{info.height} x{factor} != target {tw}x{th}")
    alpha = float((floor_blend or {}).get(factor, 1.0))
    if device.startswith("cuda"):
        try:
            return _process_item_gpu(model, input_path, output_path, device, encoder, quality, chunk, overlap,
                                     info, tw, th, factor, alpha, prefetched=prefetched, defer_finish=defer_finish)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_oom(exc):
                raise
            torch.cuda.empty_cache()
            chunk = max(4, chunk // 2)
            print(f"{input_path.name}: CUDA out of memory in the GPU pipeline, retrying on the basic path "
                  f"with chunk {chunk}", file=sys.stderr)
    t0 = time.perf_counter()
    frames = np.stack(list(iter_rgb(input_path, info.width, info.height, chroma=MODEL_INPUT_CHROMA)))
    t1 = time.perf_counter()
    while True:
        try:
            up = upscale_frames(model, frames, factor, device, chunk=chunk, overlap=overlap)
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_oom(exc):
                raise
            torch.cuda.empty_cache()
            if chunk <= 4:
                raise
            chunk = max(4, chunk // 2)
            print(f"{input_path.name}: CUDA out of memory, retrying with chunk {chunk}", file=sys.stderr)
    t2 = time.perf_counter()
    with ContractEncoder(output_path, tw, th, Fraction(info.fps), encoder=encoder, quality=quality) as enc:
        if alpha >= 1.0:
            for f in up:
                enc.write(f)
        else:
            floors = iter_i420(input_path, tw, th, vf=f"scale={tw}:{th}:flags=lanczos")
            for f, fl in zip(up, floors):
                fl = fl.astype(np.float32)
                mixed = fl + alpha * (rgb_to_i420(f).astype(np.float32) - fl)
                enc.write_i420(np.clip(np.rint(mixed), 0, 255).astype(np.uint8))
            if enc.frames != len(up):
                raise RuntimeError(f"{input_path.name}: floor decode gave {enc.frames} frames, model {len(up)}")
    t3 = time.perf_counter()
    return {"frames": int(frames.shape[0]), "decode_s": t1 - t0, "model_s": t2 - t1, "encode_s": t3 - t2,
            "total_s": t3 - t0, "spf": (t3 - t0) / max(1, frames.shape[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--encoder", default="libx264", choices=["libx264", "h264_nvenc", "hevc_nvenc"])
    ap.add_argument("--quality", type=int, default=10, help="CRF (libx264) or QP (nvenc)")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--task", default=None, help="task JSON path (default: the hidden sidecar next to the input)")
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    model = load_checkpoint(args.checkpoint, args.device)
    task = json.load(open(args.task)) if args.task else None
    stats = process_item(model, Path(args.input), Path(args.output), args.device, args.encoder, args.quality,
                         args.chunk, args.overlap, task)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
