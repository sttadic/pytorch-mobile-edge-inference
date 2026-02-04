import torch
from torchvision import models

def main():
  # Load pretrained MobileNet-V2, set to evaluation mode and cpu
  model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
  model.eval()
  model.to("cpu")

  dummy_input = torch.randn(1, 3, 224, 224)

  # Trace the model
  traced_model = torch.jit.trace(model, dummy_input)

  # Save TorchScript model to disk
  output_path = "models/mobilenet_v2_traced.pt"
  traced_model.save(output_path)

  print(f"TorchScript model saved to: {output_path}")


if __name__ == "__main__":
  main()
