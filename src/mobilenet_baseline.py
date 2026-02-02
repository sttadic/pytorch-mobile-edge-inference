import torch
from torchvision import models

def main():
  # Load pretrained MobileNet-V2
  model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

  # Switch to evaluation mode and set it to cpu (edge devices are CPU bound)
  model.eval()
  model.to("cpu")

  # Create a dummy input tensor (simulates a real image)
  dummy_input = torch.randn(1, 3, 224, 224)

  # Run a forward pass without tracking gradients
  with torch.no_grad():
    output = model(dummy_input)

  print("Output shape:", output.shape)


if __name__ == "__main__":
  main()
