import os
import torch
from torchvision import models
from preprocess import load_image

def prepare_model_for_static_quant(model):
  # Fuse Conv+BN+ReLU layers (required for static quantization)
  model.fuse_model()

  # Specify quantization config
  model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

  # Prepare model for calibration
  torch.quantization.prepare(model, inplace=True)

def calibrate(model, calibration_dir):
  print("Calibrating with real images...")
  for filename in os.listdir(calibration_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
      continue

    img_path = os.path.join(calibration_dir, filename)
    input_tensor = load_image(img_path)

    with torch.no_grad():
      model(input_tensor)

def main():
  # Load pretrained MobileNet-V2
  model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
  model.eval()

  # Prepare for static quantization
  prepare_model_for_static_quant(model)

  # Calibrate
  calibration_dir = "images/calibration"
  calibrate(model, calibration_dir)

  # Convert to quantized model
  quantized_model = torch.quantization.convert(model, inplace=False)

  # Trace and save as TorchScript
  dummy_input = torch.randn(1, 3, 224, 224)
  traced_quant = torch.jit.trace(quantized_model, dummy_input)

  output_path = "models/mobilenet_v2_static_quant.pt"
  traced_quant.save(output_path)

  print(f"Static quantized model saved to: {output_path}")


if __name__ == "__main__":
  main()

