"""Batch entry point: all items of a batch in one process (one model load per run.sh call).

    python3 upscale.py --checkpoint /app/model.pth INPUT_DIR OUTPUT_DIR

Per item: read the hidden task file, decode, upscale (x2 via pixel-unshuffle, x4 native),
encode at high quality with BT.709 tags, write atomically under the input's name. A
failure on one item is logged and the next item continues; the time budget is tracked so
the batch never overruns the timeout (a timed-out batch scores zero and counts in full).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from vidaio_train.ffio import probe  # noqa: E402
from vidaio_train.infer import SIDECAR_PREFIX, load_factor_models, process_item, read_input_i420  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--budget-seconds", type=float, default=3000.0)
    ap.add_argument("--encoder", default=os.environ.get("VIDAIO_ENCODER", "libx264"))
    ap.add_argument("--quality", type=int, default=int(os.environ.get("VIDAIO_QUALITY", "10")))
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("VIDAIO_CHUNK", "128")))
    # Shrink toward the validator's Lanczos floor per factor (see vidaio_train.infer.process_item);
    # the values baked into model.pth ("floor_blend") win over these defaults.
    ap.add_argument("--blend2", type=float, default=float(os.environ.get("VIDAIO_BLEND2", "1.0")))
    ap.add_argument("--blend4", type=float, default=float(os.environ.get("VIDAIO_BLEND4", "1.0")))
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    args = ap.parse_args()
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = os.environ.get("VIDAIO_CUDNN_BENCHMARK", "0") == "1"
    models = load_factor_models(args.checkpoint, device)   # {2: ..., 4: ...}; may be one shared model
    ck_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    blend = {2: args.blend2, 4: args.blend4}
    blend.update({int(k): float(v) for k, v in (ck_meta.get("floor_blend") or {}).items()})
    # x264 CRF per factor (the encode is measured too: crf 10 -> 6 gained ~0.015 on clean x2 items)
    crf = {2: args.quality, 4: args.quality}
    crf.update({int(k): int(v) for k, v in (ck_meta.get("encode_crf") or {}).items()})
    del ck_meta
    print(f"floor blend: {blend} | encode crf: {crf}", file=sys.stderr)
    print(f"model loaded on {device} in {time.perf_counter() - start:.1f} s", file=sys.stderr)
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    items = sorted(p for p in in_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    done = failed = 0
    # Two overlaps across items (the whole run.sh call is timed): the next input is decoded in a
    # thread while the GPU works on the current one, and the current item's encoder drains while
    # the next item's GPU work runs (deferred finish). At most one item is pending at any time.
    pool = ThreadPoolExecutor(max_workers=1)
    prefetch: dict = {}

    def _prefetch(idx: int):
        if idx < len(items):
            prefetch[idx] = pool.submit(read_input_i420, items[idx], probe(items[idx]))

    pending = None   # (item name, partial path, finish callable)

    def _settle(p) -> None:
        nonlocal done, failed
        name, partial, finish = p
        try:
            stats = finish()
            os.chmod(partial, 0o644)
            os.replace(partial, out_dir / name)
            done += 1
            print(f"done {name}: {stats['frames']} frames, {stats['total_s']:.1f} s ({stats['spf'] * 1000:.0f} ms/frame, "
                  f"gpu {stats['model_s'] / stats['frames'] * 1000:.0f} ms/frame)", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - one bad item must not abort the batch
            failed += 1
            print(f"FAILED {name}: {exc!r}", file=sys.stderr)
            try:
                partial.unlink()
            except OSError:
                pass

    _prefetch(0)
    for i, item in enumerate(items):
        left = args.budget_seconds - (time.perf_counter() - start)
        if left < 60:
            print(f"skip {item.name}: budget exhausted", file=sys.stderr)
            continue
        sidecar = in_dir / f"{SIDECAR_PREFIX}{item.name}"
        partial = out_dir / f".partial-{item.name}"
        try:
            task = json.load(open(sidecar))
            frames = prefetch.pop(i).result() if i in prefetch else None
            _prefetch(i + 1)
            result = process_item(models[int(task["upscale_factor"])], item, partial, device, args.encoder,
                                  crf[int(task["upscale_factor"])], args.chunk, 8, task,
                                  floor_blend=blend, prefetched=frames, defer_finish=True)
            if isinstance(result, tuple):
                _, finish = result
            else:                                   # CPU/fallback path finished synchronously
                finish = (lambda r=result: r)
            if pending is not None:
                _settle(pending)
            pending = (item.name, partial, finish)
        except Exception as exc:  # noqa: BLE001 - one bad item must not abort the batch
            failed += 1
            print(f"FAILED {item.name}: {exc!r}", file=sys.stderr)
            try:
                partial.unlink()
            except OSError:
                pass
            if device == "cuda":
                torch.cuda.empty_cache()
    if pending is not None:
        _settle(pending)
    pool.shutdown(wait=False)
    print(f"batch: {done} done, {failed} failed, {time.perf_counter() - start:.1f} s total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
