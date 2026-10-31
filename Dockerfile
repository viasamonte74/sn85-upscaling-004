# Contender image for sn85-upscaling-004.
# Builder rules (docs/MINING.md): plain FROM/COPY/RUN/ENV/WORKDIR only, no ADD, no tag+digest
# together, every download pinned by content, image <= 20 GB, builds in 30 min on linux/amd64.
#
# Base: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime; digest read from the Docker Hub registry manifest on
# 2026-10-29 (re-check: docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime && docker inspect --format='{{index .RepoDigests 0}}' ...).
FROM pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

# ffmpeg + ffprobe 9.0 static (libx264, same build the VidAIO scorer decodes with), pinned by digest.
COPY --from=mwader/static-ffmpeg@sha256:b90574a4e2ae62b763c39c384526689e7eb435da6398f4fb3f6c3f1c6a14ce33 /ffmpeg /ffprobe /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

# Exact-version pip installs (the precheck flags unpinned ones). torch/torchvision come with the base image.
RUN pip install --no-cache-dir numpy==2.1.3 opencv-python-headless==4.10.0.84

WORKDIR /app
COPY run.sh upscale.py /app/
COPY vidaio_train /app/vidaio_train
# Model weights ship in the repository (59 MB, two per-factor weight sets); nothing is downloaded.
COPY model.pth /app/model.pth
RUN chmod 0755 /app/run.sh && chmod 0644 /app/upscale.py /app/model.pth

# Fail the build, not the evaluation, when something the run needs is missing.
RUN ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
 && ffprobe -version >/dev/null \
 && command -v timeout >/dev/null && command -v du >/dev/null && command -v date >/dev/null \
 && python3 -c "import torch, torchvision, cv2, numpy; from vidaio_train.infer import load_factor_models; m = load_factor_models('/app/model.pth', 'cpu'); assert set(m) == {2, 4} and m[2] is not m[4], 'per-factor weights missing'; ck = torch.load('/app/model.pth', map_location='cpu', weights_only=False); assert set(ck['floor_blend']) == {2, 4} and set(ck['encode_crf']) == {2, 4}"
