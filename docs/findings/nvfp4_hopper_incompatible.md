# NVFP4 cannot run on Hopper (SM 9.0) — architectural

**TL;DR:** This artifact requires Blackwell hardware (datacenter B100/B200/B300 at SM 10.x, consumer RTX PRO 6000 / DGX Spark at SM 12.x). It will not load on H200, H100, or any Hopper-class GPU. The failure is at the kernel level, not configurable.

## Why

NVFP4 4-bit floating-point weights use NVIDIA's [tcgen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-and-related-instructions) (5th-generation tensor core / tensor-memory) instructions that were introduced with SM 10.0 (Blackwell datacenter). The vLLM NVFP4 MoE kernel — and llm-compressor's `NVFP4QuantizationModifier` save path — compile against these instructions exclusively. There is no SM 9.0 fallback path.

| Architecture | SM | NVFP4 (this artifact) | W4A16 (sibling) |
|---|---|---|---|
| H100 / H200 (Hopper) | 9.0 | ✗ no tcgen05 | ✓ Marlin MoE |
| B100 / B200 / B300 (Blackwell datacenter) | 10.0 / 10.0a | ✓ native | ✓ Marlin MoE |
| RTX PRO 6000 / DGX Spark / GB10 (Blackwell consumer) | 12.0 / 12.1 | ✓ native | ✓ Marlin MoE (with sm12x cubins, vllm#40923) |

If your hardware is Hopper, use the W4A16 sibling artifacts instead:

- [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) — base W4A16 recipe, no MTP
- [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) — W4A16 with BF16 MTP retained, gives ~1.5× spec-decode speedup at bs=1 on H200

Same recipe topology (mixed-precision FP8-attention + 4-bit experts); just Marlin instead of NVFP4 for the routed expert weights. Quality is comparable on knowledge benchmarks; NVFP4 retains a small edge on math-heavy harder benchmarks like MMLU-Pro because of less per-tensor quantization noise.

## How the failure presents (informational — don't burn GPU time to verify)

The NVFP4 MoE kernel path key-errors during the first forward pass on Hopper:

```
RuntimeError: NVFP4 quantized weights require SM 10.0+ (tcgen05 instructions)
              available on Blackwell (B100/B200/B300) or SM 12.x consumer Blackwell.
              Current device: NVIDIA H200 (SM 9.0).
```

Equivalent failures: `compute capability 9.0 is not supported by the NVFP4 GEMM kernel`, or a `RuntimeError: CUDA error: no kernel image is available for execution on the device` from the dispatched tensor-core MMA.

This is **not** a missing-flag bug, missing-cubin bug, or compile-time arch list issue. The CUDA instructions themselves don't exist on SM 9.0; no recompile fixes them.

## Provenance

This exclusion was a recurring footgun across early DSv4 quantization sessions where agents would attempt NVFP4 on Hopper after seeing "FP4" in modern-quantization marketing material. The exclusion is hardware-architectural, not bug-shaped, and so should be documented up-front in the README rather than discovered each time.
