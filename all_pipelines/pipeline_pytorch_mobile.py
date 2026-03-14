"""
Pipeline 1 – PyTorch Mobile (.ptl)

  PyTorch MobileNet-V2 (DEFAULT weights)
  → FX static INT8 quantization (QNNPACK, per-tensor)
  → TorchScript trace
  → optimize_for_mobile
  → Lite Interpreter artifact (.ptl)

Design notes
------------
- Weights:   torchvision DEFAULT (same source as ORT pipeline).
- Backend:   QNNPACK — ARM-optimised CPU kernels, no GPU/NNAPI.
- Quant:     per-tensor symmetric weights, per-tensor affine activations.
             Per-tensor is QNNPACK's native quantization mode; it does not
             support per-channel convolution weights efficiently.
- I/O:       float32 input/output (consistent with ORT and TFLite pipelines).
- Threading: set torch.set_num_threads(1) in the benchmark runner, not here.

Benchmark-runner snippet (single-threaded):
    import torch
    torch.set_num_threads(1)
    model = torch._C._jit_get_operation  # load via torch.jit.load
"""

import os
import torch
from torchvision import models
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.mobile_optimizer import optimize_for_mobile
from preprocess_shared import load_image_nchw

CALIBRATION_DIR = "images/calibration"
OUTPUT_PATH     = "models/pytorch/mobilenet_v2_static_quantized.ptl"


def get_calibration_paths(calib_dir: str) -> list:
    paths = sorted([
        os.path.join(calib_dir, f)
        for f in os.listdir(calib_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(paths)} calibration images.")
    return paths


def calibrate(model, image_paths: list):
    model.eval()
    with torch.no_grad():
        for path in image_paths:
            x = torch.from_numpy(load_image_nchw(path))  # (1, 3, 224, 224) float32
            model(x)


def main():
    # --- Load float model ---
    float_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    float_model.eval()

    # --- QNNPACK: ARM-optimised, no NNAPI/GPU ---
    torch.backends.quantized.engine = "qnnpack"

    # --- FX static quantization (per-tensor: QNNPACK's native mode) ---
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    dummy_input = torch.randn(1, 3, 224, 224)
    prepared_model = prepare_fx(float_model, qconfig_mapping, dummy_input)

    # --- Calibrate with shared dataset ---
    image_paths = get_calibration_paths(CALIBRATION_DIR)
    print(f"Calibrating ({len(image_paths)} images)...")
    calibrate(prepared_model, image_paths)

    # --- Convert to quantized model ---
    quantized_model = convert_fx(prepared_model)

    # --- TorchScript trace ---
    traced = torch.jit.trace(quantized_model, dummy_input)

    # --- Mobile graph optimisations (conv-BN fusion, mobile kernel selection) ---
    optimized = optimize_for_mobile(traced)

    # --- Save Lite Interpreter artifact ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    optimized._save_for_lite_interpreter(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
