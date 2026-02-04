import time
import torch
from torchvision import models

def benchmark(model, dummy_input, iterations=50):
  # Warmup (to stabilise timings)
  for _ in range(10):
    with torch.no_grad():
      model(dummy_input)


  # Timing loop
  start = time.time()
  for _ in range(iterations):
    with torch.no_grad():
      model(dummy_input)

  end = time.time()

  avg_ms = (end - start) / iterations * 1000
  return avg_ms

def main():
  dummy_input = torch.randn(1, 3, 224, 224)

  # Eager model
  eager_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
  eager_model.eval()
  eager_model.to("cpu")

  eager_time = benchmark(eager_model, dummy_input)
  print(f"Eager mode avg latency: {eager_time:.2f} ms")

  # Transcript model
  ts_model_path = "models/mobilenet_v2_traced.pt"
  ts_model = torch.jit.load(ts_model_path)
  ts_model.eval()

  ts_time = benchmark(ts_model, dummy_input)
  print(f"TorchScript avg latency: {ts_time:.2f} ms")


if __name__ == "__main__":
  main()
