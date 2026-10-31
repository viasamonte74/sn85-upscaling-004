# sn85-upscaling-004 entry — faithful BasicVSR++ ×2/×4

Competition entry for the VidAIO (Bittensor SN85) faithful video-upscaling track. One MP4 per
item at the original's geometry, frame count and frame rate; scored on PSNR / SSIM / LPIPS gain
over the validator's Lanczos upscale.

## What runs

- `run.sh INPUT_DIR OUTPUT_DIR` (the competition contract) calls `upscale.py` once for the whole
  batch: the model is loaded once, every regular file in `INPUT_DIR` is an item, the target
  geometry and factor are read from the hidden `.vidaio-next-upscale-task-<name>` sidecar.
- `upscale.py` → `vidaio_train.infer.process_item`: decode with ffmpeg to 8-bit 4:2:0, convert to
  RGB on the GPU (OpenCV's BT.601 matrix, bicubic chroma), run BasicVSR++ (c64n7; ×2 through a
  pixel-unshuffle stem, ×4 native) in fp16 chunks of 128 frames with 8 frames of context, shrink
  the result toward the Lanczos floor (`floor_blend` in `model.pth`: ×2 α 0.7, ×4 α 0.85), and
  encode with libx264 CRF 10, BT.709 limited-range tags, in a writer thread. A failing item is
  logged and skipped; the batch budget is tracked so `run.sh` never overruns.
- `model.pth`: EMA weights fine-tuned from the Apache-2.0 mmagic BasicVSR++ checkpoint on
  CC0 / CC BY footage, Blender films and Wan 2.2 generations degraded like the validator's inputs
  (bilinear/bicubic downscale + x264 at web bitrates). Credits in `LICENSES.md` and
  `TRAINING_DATA.md`.

## Licences

Own code: MIT (`LICENSE`). Third-party components, model provenance and training data:
`LICENSES.md`; per-file attribution of the Wikimedia Commons and YouTube-UGC sources:
`TRAINING_DATA.md`. Nothing in the image is a copy of any footage; outputs are computed from
the served input only.

## Local test

```
mkdir in out; cp <clip>.mp4 in/$(sha256sum <clip>.mp4 | cut -c1-64)
echo '{"target_width":1920,"target_height":1080,"upscale_factor":2}' > in/.vidaio-next-upscale-task-<that name>
/bin/sh run.sh in out
```
