import time
import torch
from preprocess import load_image

def benchmark(model, input_tensor, iterations=200):
    # Warm-up
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
    fp32 = torch.jit.load("models/mobilenet_v2_traced.pt")
    fp32.eval()
    fp32_time = benchmark(fp32, input_tensor)
    print(f"FP32 TorchScript: {fp32_time:.2f} ms")

    # Dynamic Quant
    dyn = torch.jit.load("models/mobilenet_v2_dynamic_quant.pt")
    dyn.eval()
    dyn_time = benchmark(dyn, input_tensor)
    print(f"INT8 Dynamic Quant: {dyn_time:.2f} ms")

    # Static Quant (x86 optimised)
    static = torch.jit.load("models/mobilenet_v2_static_quant_fx_fbgemm.pt")
    static.eval()
    static_time = benchmark(static, input_tensor)
    print(f"INT8 Static Quant (x86_fbgemm): {static_time:.2f} ms")

     # Static Quant (ARM optimised)
    static = torch.jit.load("models/mobilenet_v2_static_quant_fx_qnnpack.pt")
    static.eval()
    static_time = benchmark(static, input_tensor)
    print(f"INT8 Static Quant (ARM_qnnpack): {static_time:.2f} ms")

if __name__ == "__main__":
    main()
