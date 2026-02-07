import os
import torch
from torchvision import models
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from preprocess import load_image

def get_calibration_images(calib_dir):
  paths = []
  for name in os.listdir(calib_dir):
    if name.lower().endswith((".jpg", ".jpeg", ".png")):
      paths.append(os.path.join(calib_dir, name))
  return sorted(paths)

def calibrate(model, image_paths):
  model.eval()
  print(f"Calibrating with {len(image_paths)} images...")
  with torch.no_grad():
    for path in image_paths:
      x = load_image(path)
      model(x)

def main():
  # Load pretrained MobileNet-V2
  float_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
  float_model.eval()

  # Set backend to QNNPACK (mobile)
  torch.backends.quantized.engine = "qnnpack"

  # Create QConfigMapping for QNNPACK
  qconfig_mapping = get_default_qconfig_mapping("qnnpack")

  # Prepare FX graph for quantization
  dummy_input = torch.randn(1, 3, 224, 224)
  prepared_model = prepare_fx(float_model, qconfig_mapping, dummy_input)

  # Calibrate
  calibration_dir = "images/calibration"
  image_paths = get_calibration_images(calibration_dir)
  calibrate(prepared_model, image_paths)

  # Convert to quantized model
  quantized_model = convert_fx(prepared_model)

  # Trace and save as TorchScript
  traced_quant = torch.jit.trace(quantized_model, dummy_input)
  output_path = "models/mobilenet_v2_static_quant_fx_qnnpack.pt"
  traced_quant.save(output_path)

  print(f"Static quantized FX model saved to: {output_path}")


if __name__ == "__main__":
  main()

