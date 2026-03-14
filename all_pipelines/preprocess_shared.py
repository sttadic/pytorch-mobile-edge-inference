"""
Shared preprocessing for all three deployment pipelines.

All pipelines normalize with identical ImageNet statistics and use the same
resize/crop logic so that calibration data is equivalent across frameworks.
The only difference between the two variants is axis layout:
  - NCHW: PyTorch and ONNX models (channels-first)
  - NHWC: TFLite models after onnx-tf conversion (channels-last)
"""

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_and_normalize(path: str) -> np.ndarray:
    """
    Returns a (224, 224, 3) float32 HWC array normalized to ImageNet stats.
    Steps mirror torchvision's default MobileNet-V2 preprocessing:
      Resize shortest edge to 256 (bilinear) → centre-crop 224 → /255 → normalize.
    """
    img = Image.open(path).convert("RGB")

    # Resize: preserve aspect ratio so the shorter side becomes 256
    w, h = img.size
    if w <= h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Centre crop to 224x224
    left = (new_w - 224) // 2
    top  = (new_h - 224) // 2
    img  = img.crop((left, top, left + 224, top + 224))

    arr = np.array(img, dtype=np.float32) / 255.0      # [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD          # ImageNet normalized
    return arr                                           # (224, 224, 3) HWC


def load_image_nchw(path: str) -> np.ndarray:
    """Returns (1, 3, 224, 224) float32 — for PyTorch and ONNX/ORT models."""
    arr = _load_and_normalize(path)
    return arr.transpose(2, 0, 1)[np.newaxis, ...]      # NCHW


def load_image_nhwc(path: str) -> np.ndarray:
    """Returns (1, 224, 224, 3) float32 — for TFLite models (onnx-tf transposes to NHWC)."""
    arr = _load_and_normalize(path)
    return arr[np.newaxis, ...]                          # NHWC
