import time
import torch
from torchvision import models
from preprocess import load_image

def benchmark(model, input_tensor, iterations=30):
  # Warmup
  for _ in range(5):
    with torch.no_grad():
      model(input_tensor)

  # Timing loop
  start = time.time()
  for _ in range(iterations):
    with torch.no_grad():
      model(input_tensor)
  end = time.time()

  avg_ms = (end - start) / iterations * 1000
  return avg_ms

def main():
  # Load image
  image_path = "images/sample.jpg"
  input_tensor = load_image(image_path)

  # Eager model
  eager_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
  eager_model.eval()
  eager_model.to("cpu")

  eager_time = benchmark(eager_model, input_tensor)
  print(f"Eager mode avg latency (real image): {eager_time:.2f} ms")

  # TorchScript model
  ts_model_path = "models/mobilenet_v2_traced.pt"
  ts_model = torch.jit.load(ts_model_path)
  ts_model.eval()

  ts_time = benchmark(ts_model, input_tensor)
  print(f"TorchScript avg latency (real image): {ts_time:.2f} ms")


if __name__ == "__main__":
  main()
