import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

def optimize_and_save(input_path, output_path):
  print(f"Loading: {input_path}")
  model = torch.jit.load(input_path)
  model.eval()

  print("Optimizing for mobile...")
  optimized = optimize_for_mobile(model)

  print(f"Saving Lite Interpreter model to: {output_path}")
  optimized._save_for_lite_interpreter(output_path)

def main():
  # One to deploy to a phone
  optimize_and_save(
    "models/mobilenet_v2_static_quant_fx_qnnpack.pt",
    "models/mobilenet_v2_static_quant_fx_qnnpack_mobile.ptl"
  )

  # Optional x86 version
  optimize_and_save(
    "models/mobilenet_v2_static_quant_fx_fbgemm.pt",
    "models/mobilenet_v2_static_quant_fx_fbgemm_mobile.ptl"
  )


if __name__ == "__main__":
  main()
