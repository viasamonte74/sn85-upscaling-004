"""ffmpeg / ffprobe helpers: probing, decoding to I420 frames, raw-plane piping, contract encoding.

The competition contract (docs/MINING.md of miner-vidaio, "Upscaling competitions"):
one MP4 per item, a single H.264/HEVC/VP9/AV1 stream, yuv420p 8-bit, exactly the target
size, the input's frame count and frame rate, tagged BT.709 limited range.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np

from .color import i420_to_rgb, rgb_to_i420

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")


@dataclass(frozen=True)
class StreamInfo:
    codec: str
    width: int
    height: int
    frames: int
    fps: Fraction
    pix_fmt: str
    duration: float


def probe(path: str | os.PathLike, count_frames: bool = True) -> StreamInfo:
    """ffprobe the first video stream. Inputs have no extension: ffprobe sniffs the container."""
    args = [FFPROBE, "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        args.append("-count_frames")
    args += [
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames,pix_fmt,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    s = info["streams"][0]
    fps = Fraction(s.get("r_frame_rate") or s.get("avg_frame_rate") or "30/1")
    frames = int(s.get("nb_read_frames") or s.get("nb_frames") or 0)
    duration = float(s.get("duration") or info.get("format", {}).get("duration") or 0.0)
    return StreamInfo(
        codec=s["codec_name"], width=int(s["width"]), height=int(s["height"]), frames=frames,
        fps=fps, pix_fmt=s.get("pix_fmt", ""), duration=duration,
    )


def _popen(args: list[str], **kw) -> subprocess.Popen:
    return subprocess.Popen(args, **kw)


def iter_i420(path: str | os.PathLike, width: int, height: int, vf: str | None = None,
              fmt: str | None = None) -> Iterator[np.ndarray]:
    """Decode a video to raw I420 frames (h*3/2, w) uint8 through an ffmpeg pipe.

    ``vf`` is an optional filter chain (e.g. a Lanczos upscale for the floor). Frames are
    requested as yuv420p so the colour path equals the scorer's canonicalization."""
    args = [FFMPEG, "-v", "error", "-nostdin"]
    if fmt:
        args += ["-f", fmt]
    args += ["-i", str(path)]
    chain = (vf + "," if vf else "") + "format=yuv420p"
    args += ["-vf", chain, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    size = width * height * 3 // 2
    proc = _popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=size * 4)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height * 3 // 2, width)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc not in (0, None) and err:
            print(f"ffmpeg decode warning (rc={rc}): {err.strip()[:400]}", file=sys.stderr)


def iter_rgb(path: str | os.PathLike, width: int, height: int, vf: str | None = None,
             chroma: str = "nearest") -> Iterator[np.ndarray]:
    """Decode to RGB (h, w, 3) uint8 in the scorer's colour space (I420 -> OpenCV BT.601).

    ``chroma="nearest"`` is the scorer's decode (targets, floors, candidates);
    ``chroma=MODEL_INPUT_CHROMA`` ("cubic") is for frames the model consumes."""
    for f in iter_i420(path, width, height, vf=vf):
        yield i420_to_rgb(f, width, height, chroma=chroma)


def read_rgb(path: str | os.PathLike, vf: str | None = None, size: tuple[int, int] | None = None,
             chroma: str = "nearest") -> np.ndarray:
    """Whole clip as (t, h, w, 3) uint8. ``size`` = (w, h) after ``vf`` if the filter resizes."""
    info = probe(path, count_frames=False)
    w, h = size if size else (info.width, info.height)
    return np.stack(list(iter_rgb(path, w, h, vf=vf, chroma=chroma)))


def lanczos_vf(width: int, height: int) -> str:
    """The validator's floor: ``scale=W:H:flags=lanczos`` on the served input."""
    return f"scale={width}:{height}:flags=lanczos"


# --------------------------------------------------------------------------- writing

_BT709_TV = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def _rawvideo_input(width: int, height: int, fps: Fraction) -> list[str]:
    """Raw I420 planes on stdin with the colour tags set on the INPUT side as well.

    Do not pipe y4m: with ffmpeg 7.1 every y4m variant (C420jpeg, C420mpeg2, C420, XCOLORRANGE)
    came out resampled (Y plane 21.5 dB from what was written, even into lossless FFV1), while
    rawvideo with matching input/output tags passes the planes through (52.7 dB at CRF 10)."""
    return [FFMPEG, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}", *_BT709_TV, "-i", "-"]


class ContractEncoder:
    """Streams RGB frames (already in the scorer's RGB space) into a contract-compliant MP4.

    The frames are converted to I420 with :func:`rgb_to_i420` (exact inverse of the scorer's
    decode) and piped as raw planes with the same colour tags on input and output, so ffmpeg
    does no conversion (verified: 52.7 dB plane fidelity at CRF 10). The stream is tagged
    BT.709 limited range as the contract asks.

    ``encoder``: ``libx264`` (CRF ``quality``) or ``h264_nvenc`` (constant QP ``quality``).
    """

    def __init__(self, out_path: str | os.PathLike, width: int, height: int, fps: Fraction, *,
                 encoder: str = "libx264", quality: int = 10, preset: str | None = None,
                 threads: int = 0, container: str = "mp4"):
        self.width, self.height = width, height
        self.frames = 0
        args = _rawvideo_input(width, height, fps)
        if encoder == "libx264":
            args += ["-c:v", "libx264", "-preset", preset or "veryfast", "-crf", str(quality),
                     "-tune", "psnr", "-x264-params", "aq-mode=0:psy=0:deblock=-1,-1"]
            if threads:
                args += ["-threads", str(threads)]
        elif encoder == "h264_nvenc":
            args += ["-c:v", "h264_nvenc", "-preset", preset or "p4", "-rc", "constqp", "-qp", str(quality),
                     "-tune", "hq", "-profile:v", "high"]
        elif encoder == "hevc_nvenc":
            args += ["-c:v", "hevc_nvenc", "-preset", preset or "p4", "-rc", "constqp", "-qp", str(quality),
                     "-tune", "hq"]
        else:
            raise ValueError(f"unknown encoder {encoder!r}")
        args += ["-pix_fmt", "yuv420p", *_BT709_TV, "-movflags", "+faststart", "-f", container, str(out_path)]
        self.proc = _popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert self.proc.stdin is not None

    def write(self, rgb: np.ndarray) -> None:
        assert rgb.shape == (self.height, self.width, 3), rgb.shape
        self.write_i420(rgb_to_i420(rgb))

    def write_i420(self, i420: np.ndarray) -> None:
        """Write one frame already in the encoder's planar I420 layout (h*3/2, w) uint8."""
        assert i420.shape == (self.height * 3 // 2, self.width), i420.shape
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(i420).tobytes())
        self.frames += 1

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encode failed (rc={rc}): {err.strip()[:800]}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            try:
                self.proc.kill()
            except Exception:
                pass


def write_ffv1(out_path: str | os.PathLike, frames: Iterator[np.ndarray], width: int, height: int,
               fps: Fraction) -> int:
    """Lossless yuv420p FFV1 Matroska master from RGB frames (HR references, decoded LR copies)."""
    args = _rawvideo_input(width, height, fps) + ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p",
                                                  *_BT709_TV, str(out_path)]
    proc = _popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    n = 0
    for rgb in frames:
        proc.stdin.write(rgb_to_i420(rgb).tobytes())
        n += 1
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffv1 write failed: {err[:500]}")
    return n


def write_png_frames(out_dir: str | os.PathLike, frames: Iterator[np.ndarray]) -> int:
    """RGB frames -> out_dir/%05d.png (lossless, random access for the data loader)."""
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for rgb in frames:
        cv2.imwrite(str(out / f"{n:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        n += 1
    return n


def have_encoder(name: str) -> bool:
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return f" {name} " in out


def require_tools() -> None:
    for tool in (FFMPEG, FFPROBE):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH")
