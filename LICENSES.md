# Third-party components

Everything this entry ships besides its own code (MIT, see `LICENSE`). All of it allows
commercial use. Update this table whenever the Dockerfile changes; the licence review
checks it against what the image actually loads at run time.

| Component | Version | Source | Licence | Location |
| --- | --- | --- | --- | --- |
| PyTorch runtime image (`pytorch/pytorch`) | 2.5.1-cuda12.4-cudnn9-runtime, pinned by digest | https://hub.docker.com/r/pytorch/pytorch | BSD-3-Clause (PyTorch), NVIDIA CUDA EULA (redistributable runtime) | base image |
| torchvision | 0.20.1 (base image) | https://github.com/pytorch/vision | BSD-3-Clause | site-packages |
| NumPy | 2.1.3 | https://numpy.org | BSD-3-Clause | site-packages |
| OpenCV (opencv-python-headless) | 4.10.0.84 | https://github.com/opencv/opencv-python | Apache-2.0 | site-packages |
| FFmpeg and ffprobe, static build `mwader/static-ffmpeg` | 9.0, pinned by digest | https://github.com/wader/static-ffmpeg | GPL-3.0-or-later (run as a separate program) | `/usr/local/bin/ffmpeg`, `/usr/local/bin/ffprobe` |
| BasicVSR++ architecture (re-implemented) and pretrained initialisation | mmagic `basicvsr_plusplus_c64n7_8x1_600k_reds4` / `c128n25_ntire_vsr` | https://github.com/open-mmlab/mmagic | Apache-2.0 | `/app/vidaio_train/archs/basicvsrpp.py`; fine-tuned into `/app/model.pth`, which may hold two fine-tuned weight sets of this same Apache-2.0 base (`params_ema` for ×2 items, `params_ema_x4` for ×4 items, chosen per item from the factor in the task file by `vidaio_train/infer.py:load_factor_models`) |
| SpyNet weights (inside the BasicVSR++ checkpoint) | mmagic `spynet_20210409-c6c1bd09` | https://github.com/open-mmlab/mmagic | Apache-2.0 (published by mmagic from its repository, which is the licence that applies per the rules). Provenance note: SpyNet (Ranjan and Black, CVPR 2017) was originally released by MPI-IS under a non-commercial licence; the weights used here are the ones mmagic re-trained/converted and publishes under Apache-2.0 as part of BasicVSR++, and they were further fine-tuned in this entry (SpyNet unfrozen after step 2000). | `/app/model.pth` |

<!-- training-data:begin -->
## Training data (the training-data rule)

Everything `/app/model.pth` was trained or tuned on allows training a model for commercial use. The pretrained initialisation is the Apache-2.0 BasicVSR++ checkpoint listed above (trained by its authors on REDS). Fine-tuning used only the datasets below; every training pair was made by `vidaio_train/data/degrade.py` from these sources, and no image or video ships in the container. No output of LTX-2.x, or of any other model whose terms restrict training on its outputs, was used to train, fine-tune or tune this entry: the only synthetic training data are Wan 2.2 generations (Apache-2.0), and the three LTX-2.5 clips of the practice pack (p08-p10) were used for evaluation only.

| Dataset | Source | Licence | Used |
| --- | --- | --- | --- |
| REDS, `train_sharp` (Nah et al., NTIRE 2019 workshop) | https://seungjunnah.github.io/Datasets/reds.html (files from the Hugging Face mirror `snah/REDS`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 240 sequences of 100 frames, 1280x720 (`val_sharp` held out for validation only) |
| Big Buck Bunny, Blender Foundation (2008), www.bigbuckbunny.org | https://media.xiph.org/BBB/BBB-1080-png/ (lossless 1080p PNG renders) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 48 windows of 100 frames |
| Elephants Dream, Blender Foundation / Netherlands Media Art Institute (2006), www.elephantsdream.org | https://media.xiph.org/ED/ED-1080-png/ (lossless 1080p PNG renders) | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/) | 48 windows of 100 frames |
| Sintel, Blender Foundation (2010), www.sintel.org | https://media.xiph.org/sintel/sintel-1080-png/ (lossless 1080p PNG renders) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 48 windows of 100 frames |
| Tears of Steel, Blender Foundation (2012), www.tearsofsteel.org | https://media.xiph.org/tearsofsteel/tearsofsteel-1080-png/ (lossless 1080p PNG renders) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 48 windows of 100 frames |
| YouTube-UGC dataset (Wang, Inguva and Adsumilli, Google; user videos published on YouTube under CC BY) | https://media.withyoutube.com/ (raw 4K excerpts from `gs://ugc-dataset`, not downloaded from YouTube) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), per video | 103 videos (Vlog, Sports, VerticalVideo, LiveMusic, HowTo, CoverSong; 2160P originals area-downscaled to 1080p, 1080P originals at native size), first 4.5 s each; per-video credits in `TRAINING_DATA.md` |
| Wikimedia Commons videos (phone, drone, action-camera and general 4K footage; CC BY-SA and non-free files excluded) | https://commons.wikimedia.org/ | per file: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) (65), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (47), [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (60), [Public domain](https://commons.wikimedia.org/wiki/Commons:Licensing#Material_in_the_public_domain) (1) | 173 videos, 347 clips; per-file title, author, link and licence in `TRAINING_DATA.md` |
| Wan 2.2 TI2V-5B generations (text-to-video outputs made for this run with `scripts/generate_wan22.py`; MINING.md lists Wan 2.x outputs as allowed training data) | https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers | model weights [Apache-2.0](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt); outputs are our own | 60 clips of 121 frames, 1280x704 / 704x1280, 24 fps; prompts and seeds in `TRAINING_DATA.md` |
| Held-out evaluation clips (never trained on) | REDS `val_sharp` (CC BY 4.0); 28 further YouTube-UGC 1080P originals (CC BY 4.0, attribution in `data/sources/ugc` of the training repository); other time segments of Glass Half and SINGULARITY (Blender, CC BY 4.0, Wikimedia Commons) | CC BY 4.0 | **evaluation only**: used to measure candidate checkpoints and inference settings; every candidate measured on them was rejected, so they did not change the shipped weights or settings |
| practice-pack-004 p01-p10 (VidAIO) | https://github.com/vidAio-subnet/miner-vidaio/releases/tag/practice-pack-004 | p01 VidAIO footage, p02-p07 CC0; p08-p10 are LTX-2.5 outputs | **validation only, never trained on.** Checkpoint selection and the only tuned inference settings (the per-factor shrink toward the Lanczos floor, `floor_blend` in `model.pth`, and the per-factor x264 CRF of the output encode, `encode_crf`, both chosen on the phone item p01 and the render items p02-p07) use p01-p07 only; p08-p10 are scored for information and never enter a decision |

Generated by `scripts/write_training_credits.py` from the source manifests: 942 training clips (bbb 48, commonsA 38, commonsB 90, commonsC 80, commonsR 28, commonsS2 111, ed 48, reds 240, sintel 48, tos 48, ugc 103, wan 60).
<!-- training-data:end -->
