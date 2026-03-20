"""
Pipeline 2 – TensorFlow Lite (.tflite)

  PyTorch MobileNet-V2 (DEFAULT weights)
  → ONNX export (opset 18, shared with ORT pipeline)
  → TF SavedModel via onnx2tf  [NCHW → NHWC transposed automatically]
  → TFLite full-integer INT8 quantization (float32 I/O, XNNPACK-ready)

Dependencies
------------
    pip install onnx2tf tensorflow

Design notes
------------
- Weights:   same PyTorch DEFAULT weights as the other pipelines (via ONNX).
             The original tensorflow_step1_export_model.py used TF's own
             ImageNet weights, which are different — that is fixed here.
- Backend:   XNNPACK is TFLite's default CPU delegate for INT8 ops on ARM.
             It is enabled automatically when using TFLITE_BUILTINS_INT8 ops;
             no GPU delegate or NNAPI is configured.
- Quant:     full-integer INT8 (weights + activations), per-channel for conv
             weights (TFLite's default with Optimize.DEFAULT).
- I/O:       float32 input/output — inference_input_type and inference_output_type
             are intentionally NOT set (defaults to float32), consistent with
             the PyTorch and ORT pipelines. The original script used INT8 I/O;
             float32 I/O is used here so all pipelines measure the same
             dequantisation boundary overhead.
- Layout:    onnx2tf transposes NCHW → NHWC and runs onnxsim (ONNX graph
             simplifier) before conversion, producing a cleaner TF graph than
             onnx-tf. Calibration images are fed as NHWC to match.
- Threading: pass num_threads=1 to tf.lite.Interpreter in the benchmark runner.

Benchmark-runner snippet (single-threaded, XNNPACK explicit):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(
        model_path="models/tflite/mobilenet_v2_static_quantized.tflite",
        num_threads=1,
    )
    interpreter.allocate_tensors()
"""

import os
import numpy as np
import torch
from torchvision import models
import onnx2tf
import tensorflow as tf
from preprocess_shared import load_image_nhwc

CALIBRATION_DIR = "images/calibration"
ONNX_PATH       = "models/onnx/mobilenet_v2_float.onnx"
SAVED_MODEL_DIR = "models/tflite/saved_model"
OUTPUT_PATH     = "models/tflite/mobilenet_v2_static_quantized.tflite"


def export_onnx(onnx_path: str):
    """Export PyTorch MobileNet-V2 to ONNX (NCHW, opset 18)."""
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


def onnx_to_saved_model(onnx_path: str, saved_model_dir: str):
    """
    Convert ONNX → TF SavedModel via onnx2tf.
    onnx2tf runs onnxsim to simplify the graph, then transposes NCHW conv ops
    to NHWC. The resulting SavedModel input is therefore (1, 224, 224, 3) NHWC.
    """
    print("Converting ONNX → TF SavedModel (onnx2tf)...")
    os.makedirs(saved_model_dir, exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        non_verbose=True,
    )
    print(f"TF SavedModel saved: {saved_model_dir}")


def get_calibration_paths(calib_dir: str) -> list:
    return sorted([
        os.path.join(calib_dir, f)
        for f in os.listdir(calib_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def representative_dataset(image_paths: list):
    """
    Yields (1, 224, 224, 3) float32 NHWC tensors.
    Layout matches the onnx-tf SavedModel's transposed input.
    Values use the same ImageNet normalisation as the other pipelines.
    """
    print(f"Calibrating with {len(image_paths)} images...")
    for path in image_paths:
        yield [load_image_nhwc(path)]  # (1, 224, 224, 3) float32


def quantize_to_tflite(saved_model_dir: str, image_paths: list, output_path: str):
    """
    Full-integer INT8 quantization with float32 I/O.

    inference_input_type and inference_output_type are intentionally left at
    their defaults (float32) so that benchmark I/O handling is identical to
    the PyTorch and ORT pipelines.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(image_paths)

    # INT8 ops throughout; float32 I/O boundaries (no INT8 I/O shortcut)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {output_path}")


def main():
    # Step 1: ONNX export (shared with ORT pipeline; skip if already exists)
    if not os.path.exists(ONNX_PATH):
        export_onnx(ONNX_PATH)
    else:
        print(f"Reusing existing ONNX model: {ONNX_PATH}")

    # Step 2: ONNX → TF SavedModel
    if not os.path.exists(SAVED_MODEL_DIR):
        onnx_to_saved_model(ONNX_PATH, SAVED_MODEL_DIR)
    else:
        print(f"Reusing existing SavedModel: {SAVED_MODEL_DIR}")

    # Step 3: TFLite INT8 quantization (float32 I/O)
    image_paths = get_calibration_paths(CALIBRATION_DIR)
    quantize_to_tflite(SAVED_MODEL_DIR, image_paths, OUTPUT_PATH)


if __name__ == "__main__":
    main()
