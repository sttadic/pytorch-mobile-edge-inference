import torch
from torchvision import models

def main():
  # Load FP32 model
  model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
  model.eval()

  # Apply dynamic quantization
  quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # only linear layers get quantized
    dtype=torch.qint8
  )

  # Save quantized model as TorchScript
  dummy_input = torch.randn(1, 3, 224, 224)
  traced_quant = torch.jit.trace(quantized_model, dummy_input)

  output_path = "models/mobilenet_v2_dynamic_quant.pt"
  traced_quant.save(output_path)

  print(f"Dynamic quantized model saved to: {output_path}")


if __name__ == "__main__":
  main()
