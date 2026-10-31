"""Training, validation and inference code for the VidAIO faithful upscaling track.

Everything that touches the score is written to mirror the validator's scorer
(``vidaio/scoring/upscaling_faithful*.py`` in the miner-vidaio repository):

* frames are compared in the RGB space that OpenCV's ``COLOR_YUV2RGB_I420`` produces
  from the 8-bit 4:2:0 planes (:mod:`vidaio_train.color`);
* PSNR over all RGB pixels, SSIM on luma with a Gaussian 11/1.5 window, LPIPS-VGG on
  frames area-downscaled to a 512 px long side (:mod:`vidaio_train.metrics`);
* the training losses are the differentiable versions of the same three terms
  (:mod:`vidaio_train.losses`).
"""

__version__ = "0.1.0"
