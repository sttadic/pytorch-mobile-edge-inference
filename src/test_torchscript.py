import torch

def main():
  # Load the TorchScript model
  model_path = "models/mobilenet_v2_traced.pt"
  model = torch.jit.load(model_path)
  model.eval()

  dummy_input = torch.randn(1, 3, 224, 224)

  # Run inference
  with torch.no_grad():
    output = model(dummy_input)

  print("TorchScript output shape:", output.shape)


if __name__ == "__main__":
  main()
