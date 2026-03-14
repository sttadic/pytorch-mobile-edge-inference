# MobileNet-V2 ARM Benchmark — Design Decisions

Last updated: 2026-03-14 (onnx2tf switch; target devices confirmed; ORT backend clarified)

---

## Research Goal

Determine which ML inference framework an engineer should choose for on-device deployment on ARM hardware.
Frameworks compared: PyTorch Mobile, TensorFlow Lite, ONNX Runtime.
Model: MobileNet-V2.

**Target devices:**
- Primary: Raspberry Pi 4B (ARM Cortex-A72, aarch64, Linux)
- Optional: Samsung Galaxy S24 FE (Android)

---

## Key Design Question: Uniform Pipeline vs Best-of-Each

### The question
Should each framework be forced through an identical quantization/optimization process (for comparability), or should each framework be allowed to apply its best available optimizations?

### Thinking process

**Uniform pipeline** would answer: "Given identical methodology, which framework's underlying infrastructure is faster?"
- Pro: controls for methodology, isolates framework runtime differences
- Con: forces frameworks into unnatural configurations; may penalize frameworks with legitimate architectural reasons for different defaults; distorts results

**Best-of-each** would answer: "In practice, which framework should a developer choose?"
- Pro: reflects real deployment reality; showcases true capabilities; directly useful to practitioners
- Con: harder to attribute differences to specific factors; less controlled

**Why the uniform approach is a false premise:**
Quantization is not framework-agnostic. INT8 PTQ in TFLite uses different schemes (symmetric/asymmetric, per-channel/per-tensor) than PyTorch's `torch.ao.quantization` or ORT's quantization tooling. Forcing them to behave identically introduces artificial distortions rather than removing them.

### Decision: Best-of-Each

Rationale: the research question is practitioner-facing ("which framework should I choose?"). Constraining each framework to an artificial common process would mislead the reader — a developer following the research would get worse results than what each framework is capable of. Best-of-each also captures the real developer experience.

- What should be uniform — things that are inputs to the process: same calibration dataset (your 50 images), same source model weights, same target quantization type (static INT8). These are controls, not constraints on the framework.                                                                                                                                   
- What should not be forced — how each framework internally implements that: quantization granularity (per-channel vs per-tensor), operator fusion, delegate selection, backend-specific passes. Let each framework decide.

---

## What "Best" Means Per Framework (as implemented)

| Framework | Quantization | Backend | Key choices |
|---|---|---|---|
| PyTorch Mobile | FX static INT8, per-tensor weights | QNNPACK | Per-tensor is QNNPACK's native mode; `optimize_for_mobile` applied |
| TFLite | Full-integer INT8, per-channel weights (default) | XNNPACK (auto) | `Optimize.DEFAULT` + `TFLITE_BUILTINS_INT8`; float32 I/O |
| ORT | QDQ static INT8, per-channel weights | MLAS/CPU on Pi 4B; XNNPACK EP on Android | `per_channel=True` — ORT's recommended QDQ setting for ARM |

**Shared inputs (legitimate controls, not constraints):**
- Same source weights (PyTorch DEFAULT)
- Same calibration dataset and preprocessing
- Static INT8 target for all
- Float32 I/O at inference boundary (consistent dequantisation overhead)
- Single-threaded benchmark (environmental control)

---

## Per-Pipeline Best-Effort Audit

A review of whether each pipeline is genuinely doing its best given the shared inputs (same weights, same calibration set, static INT8 target).

### PyTorch Mobile — satisfactory

`get_default_qconfig_mapping("qnnpack")` is the canonical PyTorch approach for ARM. Per-tensor is not a constraint — it is QNNPACK's native quantization mode; QNNPACK does not support per-channel convolution weights efficiently. `optimize_for_mobile` is applied, which performs conv-BN fusion and mobile-specific kernel selection. No issues identified.

### ONNX Runtime — satisfactory (after fix), with one open backend question

Originally used `per_channel=False` to artificially match QNNPACK's per-tensor scheme. This was a forced constraint, not ORT's best setting. Fixed to `per_channel=True`, which is ORT's recommended QDQ configuration for ARM and gives better accuracy with no runtime penalty on supported kernels. QDQ format is correct for ORT ARM kernels.

**Execution provider — resolved per device.** ORT has two relevant CPU providers on ARM:
- `CPUExecutionProvider`: uses ORT's MLAS kernels with ARM NEON optimisations
- `XNNPACKExecutionProvider`: uses Google's XNNPACK library (same backend TFLite uses automatically)

The docstring previously described `CPUExecutionProvider` as "ARM/XNNPACK" — these are distinct providers; that was incorrect.

**Raspberry Pi 4B (primary target):** The standard `onnxruntime` pip wheel for Linux aarch64 does **not** include XNNPACK EP. Obtaining it requires building ORT from source with `--use_xnnpack`, which is impractical. `CPUExecutionProvider` (MLAS + NEON) is therefore the correct and realistic best for the Pi. No change needed.

**Samsung Galaxy S24 FE (optional target):** `onnxruntime-mobile` for Android **does** include XNNPACK EP. If benchmarks are run on the S24 FE, the benchmark runner should use:

```python
providers=["XNNPACKExecutionProvider", "CPUExecutionProvider"]  # XNNPACK with CPU fallback
```

Note: `NnapiExecutionProvider` is also available on Android and can delegate to hardware accelerators (GPU, DSP, NPU), but that takes the comparison outside pure-CPU territory and would be a separate experiment.

**Summary:** ORT's backend will differ between devices — MLAS on Pi 4B, XNNPACK on S24 FE. This is correct behaviour for best-of-each: each device gets ORT's best available CPU backend for that platform. Document clearly in the results section.

### TFLite — satisfactory (after conversion toolchain fix)

The quantization settings are correct: `Optimize.DEFAULT` + `TFLITE_BUILTINS_INT8` is TFLite's recommended full-integer INT8 approach, and per-channel weights for conv layers are applied automatically.

**Conversion toolchain concern (resolved):** The original pipeline used `onnx-tf` for the ONNX → TF SavedModel step:

```
PyTorch → ONNX → onnx-tf → TF SavedModel → TFLite   # original, replaced
```

`onnx-tf` is poorly maintained and produces suboptimal TF graphs — extra transpose ops, non-fused patterns, compatibility issues. Because TFLite's quantization quality depends on the graph it receives, a suboptimal SavedModel can penalise TFLite's benchmark results due to the toolchain, not the framework.

**Fixed: switched to `onnx2tf`**:

```
PyTorch → ONNX → onnx2tf → TF SavedModel → TFLite   # current
```

`onnx2tf` is actively maintained, runs `onnxsim` (ONNX graph simplifier) before conversion to produce a cleaner intermediate graph, and handles NCHW → NHWC transposes more reliably. The TFLite converter step (including `representative_dataset`, `Optimize.DEFAULT`, `TFLITE_BUILTINS_INT8`) is unchanged — only the `onnx_to_saved_model()` function was updated.

**Future note:** The onnx2tf maintainer flagged (May 2024) that AI Edge Torch — Google's native PyTorch → TFLite path with no ONNX intermediary — may eventually supersede it. Not a concern for this research, but worth revisiting if the work is extended.

---

## Metrics Planned

| Metric | Notes |
|---|---|
| Latency | Warm-up runs required; control CPU governor and thread affinity |
| Model size | Compressed footprint matters for storage/memory-constrained devices |
| CPU utilisation | Per-framework threading behaviour may differ significantly |
| Thermal parameters | Important for sustained inference workloads on ARM |
| Developer experience | Qualitative; reflects ease of adoption |

**Important:** accuracy degradation post-quantization must also be reported alongside latency. A 2x speedup with a 5% accuracy drop is a meaningful tradeoff, not a win.

---

## Environment Controls (ARM Device)

To ensure results are attributable to frameworks and not environment noise:
- Pin CPU governor (e.g., `performance` mode)
- Control thread affinity
- Define number of warm-up runs and averaged inference runs
- Keep thermal state consistent across runs (thermal parameters being measured, so document starting conditions)

---

## Open Questions / Next Steps

- [ ] Finalize ARM device and OS version
- [ ] Define calibration dataset and number of samples for PTQ
- [ ] Define accuracy metric (Top-1 on ImageNet validation subset?)
- [ ] Define developer experience rubric (time to deploy, API complexity, debugging difficulty, documentation quality)
- [x] Switch TFLite pipeline from onnx-tf to onnx2tf (done — see Per-Pipeline Audit)
- [x] ORT execution provider resolved per device: CPUExecutionProvider (MLAS) on Pi 4B; XNNPACKExecutionProvider on S24 FE if benchmarked (see Per-Pipeline Audit)

---

## Additinal Notes
> Question:
 Is current implementation of all pipelines correct provided benchmarks are run only on Raspberry Pi 4b?                                                                           

● Yes, all three pipelines are correct as-is for Raspberry Pi 4B. To summarise why each one holds up:

  - PyTorch Mobile — QNNPACK is the right ARM backend, per-tensor is its native mode, optimize_for_mobile is applied. Nothing to change.
  - TFLite — Optimize.DEFAULT + TFLITE_BUILTINS_INT8 is correct. Importantly, XNNPACK is available in the standard TensorFlow pip wheel on Linux ARM, so TFLite gets its optimal delegate
  on the Pi without any extra steps. onnx2tf handles the conversion cleanly.
  - ORT — CPUExecutionProvider (MLAS + NEON) is exactly the right choice for Linux aarch64, since XNNPACK EP isn't in the standard pip wheel. This is ORT's genuine best on the Pi.

  The only thing that would need touching if you later add the S24 FE is the ORT benchmark runner's providers list — and that's already documented. The quantization pipelines themselves
  wouldn't change at all.

  So for Pi 4B only we are good to go.