"""
Pipeline 3 – ONNX Runtime Mobile (.onnx, INT8)

  PyTorch MobileNet-V2 (DEFAULT weights)
  → ONNX export (opset 18, shared with TFLite pipeline)
  → ORT static INT8 quantization (QDQ format, per-channel weights)

Dependencies
------------
    pip install onnx onnxruntime

    For ARM deployment: use the onnxruntime-mobile wheel on the target device.

Design notes
------------
- Weights:   same PyTorch DEFAULT weights as the other pipelines.
             The original codebase had no ORT quantization step at all —
             it benchmarked a float32 model against two INT8 models.
- Backend:   ORT's built-in ARM/XNNPACK execution provider; no GPU EP,
             no CoreML, no NNAPI.
- Quant:     QDQ format, per-channel weights (per_channel=True) — ORT's
             recommended setting for ARM; gives better accuracy than per-tensor
             with no runtime penalty on supported kernels.
             Activation type: QUInt8. Weight type: QInt8.
- I/O:       float32 input/output (consistent with PyTorch and TFLite pipelines).
- Path:      models/onnx/mobilenet_v2_float.onnx — shared with TFLite pipeline.
             The original codebase exported to _old/models/ but loaded from
             models/, causing a path mismatch. Fixed here with a single path.
- Threading: set intra_op_num_threads=1 in the benchmark runner (see below).

Benchmark-runner snippet (single-threaded):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        "models/ort/mobilenet_v2_static_quantized.onnx",
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
"""

import os
import numpy as np
import torch
from torchvision import models
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
from preprocess_shared import load_image_nchw

CALIBRATION_DIR     = "images/calibration"
ONNX_FLOAT_PATH     = "models/onnx/mobilenet_v2_float.onnx"
ONNX_QUANTIZED_PATH = "models/ort/mobilenet_v2_static_quantized.onnx"


def export_onnx(onnx_path: str):
    """Export PyTorch MobileNet-V2 to ONNX (NCHW, opset 12)."""
    print("Exporting PyTorch model to ONNX...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"ONNX model saved: {onnx_path}")


def get_calibration_paths(calib_dir: str) -> list:
    return sorted([
        os.path.join(calib_dir, f)
        for f in os.listdir(calib_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


class MobileNetCalibrationReader(CalibrationDataReader):
    """
    Feeds (1, 3, 224, 224) NCHW float32 images to the ORT quantizer.
    Uses the same ImageNet normalisation as the PyTorch and TFLite pipelines.
    """

    def __init__(self, image_paths: list, input_name: str = "input"):
        self.image_paths = image_paths
        self.input_name  = input_name
        self._index      = 0
        print(f"Calibrating with {len(image_paths)} images...")

    def get_next(self) -> dict | None:
        if self._index >= len(self.image_paths):
            return None
        arr = load_image_nchw(self.image_paths[self._index])  # (1, 3, 224, 224) float32
        self._index += 1
        return {self.input_name: arr}


def main():
    # Step 1: ONNX export (shared with TFLite pipeline; skip if already exists)
    if not os.path.exists(ONNX_FLOAT_PATH):
        export_onnx(ONNX_FLOAT_PATH)
    else:
        print(f"Reusing existing ONNX model: {ONNX_FLOAT_PATH}")

    # Step 2: Static INT8 quantization
    image_paths = get_calibration_paths(CALIBRATION_DIR)
    os.makedirs(os.path.dirname(ONNX_QUANTIZED_PATH), exist_ok=True)

    quantize_static(
        model_input=ONNX_FLOAT_PATH,
        model_output=ONNX_QUANTIZED_PATH,
        calibration_data_reader=MobileNetCalibrationReader(image_paths),
        quant_format=QuantFormat.QDQ,       # QDQ: recommended for ORT ARM kernels
        per_channel=True,                   # per-channel weights: ORT's recommended QDQ setting
        weight_type=QuantType.QInt8,        # signed INT8 weights
        activation_type=QuantType.QUInt8,   # unsigned INT8 activations
    )
    print(f"Saved: {ONNX_QUANTIZED_PATH}")


if __name__ == "__main__":
    main()
