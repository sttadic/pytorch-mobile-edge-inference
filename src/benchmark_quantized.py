import time
import torch
from torchvision import models
from preprocess import load_image

def benchmark(model, input_tensor, iterations=30):
  for _ in range(5):
    with torch.no_grad():
      model(input_tensor)

  start = time.time()
  for _ in range(iterations):
    with torch.no_grad():
      model(input_tensor)
  end = time.time()


  return (end - start) / iterations * 1000

def main():
  image_path = "images/sample.jpg"
  input_tensor = load_image(image_path)

  # FP32 TorchScript
  ts_model = torch.jit.load("models/mobilenet_v2_traced.pt")
  ts_model.eval()
  ts_time = benchmark(ts_model, input_tensor)
  print(f"FP32 TorchScript latency: {ts_time:.2f} ms")

  # INT8 Dynamic Quantized TorchScript
  quant_model = torch.jit.load("models/mobilenet_v2_dynamic_quant.pt")
  quant_model.eval()
  quant_time = benchmark(quant_model, input_tensor)
  print(f"INT8 Dynamic Quant latency: {quant_time:.2f} ms")


if __name__ == "__main__":
  main()
