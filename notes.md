Before comparing PyTorch Mobile, ONNX Runtime, and TensorFlow Lite, I’ll first compare eager PyTorch and TorchScript
models to validate the pipeline before going mobile. This provides a baseline in case issues arise later (quantization,
mobile runtimes, ONNX export, etc.), making it easier to pinpoint where problems occur and to sanity-check performance.

# Loading MobileNet‑V2 in Eager mode | Eager vs TorchScript
> src/mobilenet_baseline.py

Loading MobileNet‑V2 (pretrained) in eager mode (normal PyTorch). \
Eager mode is the default way to use PyTorch - write a model in Python, and it runs immediately.

**Characteristics of an eager model (mode):**
- Written in Python (nn.Module, forward, etc.)
- Uses dynamic control flow (loops, if‑statements, Python objects)
- Builds the computation graph on the fly during execution
- Great for debugging and experimentation
- Requires the Python interpreter to run

PyTorch executes operation eagerly. There is no compilation step, no static graph, no ahead-of-time
optimization.
So eager mode depends on Python interpreter, it's slower to start up, can't be easily optimised, can't run
inside Androind/iOS apps (they don't ship Python).

Eager model is just the uncompiled, dynamic (can change at runtime), Python-dependent version.
TorchScript model is the compiled, static (doesn't change at runtime), Python-free version.
TorchScript is also serialized (converted into a binary file format that can be saved to disk and loaded)
version of your model.

**This file will:**
- Load MobileNet‑V2
- Switch it to evaluation mode
- Run it on CPU (matches edge-device scenario)
- Create a dummy input: batch size = 1, channels = 3 (RGB), hieght = 224, width = 224
- Run a single inference on a dummy input (feed the dummy tensor through the model)
- Print the output shape (dimensions of tensor that comes out of the model's forward pass: size & structure)

**Additional note:**\
Output shape: [1, 1000] indicates that a batch of one image was passed to the model. MobileNet-V2 is trained on
ImageNet, which contains 1000 classes, so the model outputs a vector of 1000 values—one score per class. These
values are logits, i.e. raw, unnormalized scores, which are typically converted to probabilities (for example,
using softmax).

# Exporting to TorchScript
> src/export_torchscript.py

Here is where PyTorch becomes a deployable artifact.

**There are two ways to turn PyTorch model into TorchScript (.pt):**
* Tracing (good for models with no dynamic control flow, standard CNNs like MobileNet‑V2...)
   - You give the model a sample input (like your dummy tensor).
   - PyTorch runs the model once.
   - It records all operations that happened.
   - The result is a static computation graph.
* Scripting (good for complex models with dynamic behaviour e.g. with if and for loops, custom layers...)
   - PyTorch reads the model’s Python code.
   - It converts it into TorchScript directly.
   - It understands control flow (if, for, loops).

I'll use tracing since MobileNet‑V2 is pure feed-forward CNN with no dynamic control flow.

**This file will:**
- Load MobileNet‑V2 (same as before)
- Create a dummy input (same as before)
- Trace the model
- Save the TorchScript file to disk

# Testing TorchScript model
> src/test_torchscript.py

Loading TorchScript model and running inference (sanity check).

The goal here is to make sure exported TorchScript file is valid, that it produces output with correct
shape (same as eager model), and to verify TorchScript inference works end-to-end on CPU.

So far I confirmed that Eager and TorchScript model work, they produce the same output shape, and both
run on CPU. This means pipeline is stable and ready for benchmarking, quantization, model size comparison,
memory profiling.

# Benchmarking Eager vs TorchScript
> src/benchmark_basic.py

Compare eager vs TorchScript latency (basic latency comparison).\
This step gives first real performance numbers.

**I'll keep it simple:**
- One script
- Two timing loops
- Same dummy input
- Compare average inference time
- No optimizations yet. Just a clean, honest baseline.

**TorchScript is usually 10–30% faster on CPU for MobileNet‑V2, but exact numbers depend on:**
- CPU frequency scaling
- Background processes
- WSL2 overhead
- Python version

**Benchmarking results:**
- Eager mode: 24.75ms
- TorchScript: 16.21ms

# Adding Image Preprocessing
> src/preprocess.py

Before going mobile, I need to add real image preprocessing.\
Up to now I used dummy tensors (random numbers) as input. Now I’ll load a real image from disk,
apply standard ImageNet preprocessing, and feed it to both eager and TorchScript models.

**This step introduces:**
- Loading a real image
- Resizing to 224×224
- Converting to tensor
- Normalizing using ImageNet stats
- Preparing input for both eager and TorchScript models

``` python
# Standard ImageNet preprocessing
preprocess = transforms.Compose([
  # Resize shortest side to 256 pixels with aspect ratio preserved
  transforms.Resize(256),
  # Crop the center 224x224 pixels
  transforms.CenterCrop(224),
  # Convert PIL Image to PyTorch tensor (C x H x W) in range [0.0, 1.0]
  transforms.ToTensor(),
  # Normalize using ImageNet mean and std
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )
])

def load_image(path):
  with Image.open(path).convert("RGB") as img:
    tensor = preprocess(img)
# Adds a new dimension at index 0, turning the tensor from shape [3, 224, 224] into [1, 3, 224, 224]. Models in PyTorch expect a batch dimension, even for a single image.
  tensor = tensor.unsqueeze(0)  # add batch dimension
  return tensor
```
I now have a reusable function **load_image("path/to/image.jpg")** which returns a properly normalized image tensor ready for Eager and TorchScript models, quantizied models, benchmarking, mobile runtimes, etc.

# Benchmarking Eager vs TorchScript with Real Image
> src/benchmark_real_image.py

Now that I have real image preprocessing, I can benchmark Eager vs TorchScript using a real image instead of dummy tensors.\This gives a more realistic performance comparison, since real image preprocessing can affect latency.

**This step will:**
- Load a real image from disk
- Preprocess it using preprocess.py module
- Run inference on both eager and TorchScript models
- Measure latency for each
- Print results

**Benchmarking results:**
- Eager mode: 16.52ms
- TorchScript: 12.40ms

# Dynamic Quantization (INT8)
> src/quantize_dynamic.py

This is the first and easiest mobile optimization technique. It can be applied to the existing TorchScript model without changing the architecture or retraining.

**Dynamic quantization will give the following benefits:**
- Smaller model size (up to 4× smaller)
- Faster inference on CPU (especially on mobile/edge devices - ARM CPUs with limited compute power)
- Minimal accuracy loss (usually less than 1% for image classification)

Dynamic quantization is a technique that reduces model size and can improve inference speed by converting weights and activations from 32-bit floating point (FP32) to 8-bit integers (INT8).\
This is especially beneficial for CPU inference, which is common in mobile and edge devices.

**Size comparison:**
- FP32 TorchScript: 13.88MB
- INT8 Dynamic Quantized: 10.22MB

After applying dynamic quantization to the MobileNet‑V2 model, model size is reduced from 13.88MB to 10.22MB which is less than expected (4× smaller would be around 3.5MB). This is likely because dynamnic quantization only quantizes nn.Linear layers, and MobileNet‑V2 has a lot of convolutional layers which remain in FP32.

**Benchmarking results:**
- FP32 TorchScript: 14.12ms
- INT8 Dynamic Quantized: 12.01ms

It's worth mentioning that on ARM CPUs, the speedup would be even more significant, often 2–4× faster than FP32. The 15% speedup observed here on desktop CPU is a good sign that quantization is working, but the real benefits will be seen on mobile devices.

# What is achieved so far?
- FP32 eager model
- FP32 TorchScript model
- INT8 dynamic quantized TorchScript model
- Real‑image preprocessing
- Real‑image benchmarking
- Model size comparison
- Latency comparison

# Static Quantization (INT8)
> src/quantize_static.py
