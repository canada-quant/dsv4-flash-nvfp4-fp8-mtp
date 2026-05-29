---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash
tags:
  - compressed-tensors
  - nvfp4
  - fp8
  - vllm
  - deepseek
  - mtp
  - speculative-decoding
  - mixture-of-experts
library_name: vllm
---

# canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP

A DeepSeek-V4-Flash NVFP4-FP8 quantization that retains the MTP (multi-token-prediction) block in the saved weights, so vLLM can load it with `--speculative-config method=mtp`.

> ## 🛑 Hardware scope — B200 / B300 only
>
> This artifact targets **datacenter Blackwell (SM 10.0 / sm_100a / sm_103a)** — the only hardware where vLLM's native NVFP4 tensor-core path (`tcgen05`) actually executes the MoE matmul in FP4.
>
> **On consumer Blackwell (RTX PRO 6000 SE / DGX Spark / RTX 5090, SM 12.0+)** vLLM falls back to the Marlin BF16 kernel which dequantizes FP4 → BF16 inside the kernel. The fallback works correctly but **forfeits the FP4 FLOPS advantage** that motivates NVFP4 in the first place, and on the same RTX PRO 6000 hardware the [W4A16 sibling](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) has a smaller footprint (~159 GB vs 172 GB) and slightly higher throughput on its native Marlin INT4 path.
>
> **If you're deploying on RTX PRO 6000, DGX Spark, RTX 5090, or any consumer Blackwell, use the W4A16 sibling instead:**
>
> - [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP)

## What this is

- 172 GB across 35 safetensors shards (vs ~600 GB BF16 source, MTP block included).
- Same quantization scheme as `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`: NVFP4 (group=16, FP8 e4m3 scales) on routed FFN experts, FP8_BLOCK 128×128 on attention.
- MTP block (`mtp.0.*`, 799 tensors) kept at BF16 — not dropped at load time, not double-quantized when the MTP draft model is constructed.

That last point is the only structural difference from RedHat's artifact. The HF transformers DSV4 modeling class has `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`, which silently strips MTP keys during the calibration load path. RedHat's artifact ran through that path, so their saved weights don't include MTP. Ours include MTP because we patched the modeling class during calibration.

## Hardware validated

| Platform | Compute cap | HBM per GPU | Interconnect | Role |
|---|---|---|---|---|
| 4× NVIDIA B300 SXM6 AC | SM 10.3 (`sm_103a`) | 288 GB HBM3e | NVLink | **Primary** — all accuracy + throughput numbers below; native FP4 tensor cores active |

Server with CUDA graphs enabled. RTX PRO 6000 / DGX Spark / RTX 5090 are out of scope for this artifact — use the [W4A16 sibling](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) for those.

## Accuracy (4× B300, TP=4)

Measured 2026-05-21. Same prompts, temperature 0, chat template server-side.

| Benchmark | This artifact | BF16 + MTP reference (TP=8) | RedHat (no MTP) |
|---|---|---|---|
| AIME 2024 raw pass@1 (thinking=high, max_tokens=65536) | 25/30 = 83.33% | 25/30 = 83.33% | 27/30 = 90.00% |
| AIME 2024 non-truncated pass@1 | 24/25 = 96.00% | 25/26 = 96.15% | 27/28 = 96.43% |
| AIME 2024 wall-clock (30 problems, c=8) | 476s | 490s | 1405s |
| MTP draft acceptance, AIME reasoning | 81.60% | 78.19% | n/a |
| GSM8K strict-match (8-shot) | 0.9181 | 0.9484 / 0.9522 (no-MTP / MTP) | 0.910 (self-reported) |
| GSM8K flexible-extract (8-shot) | 0.9515 | 0.9477 / 0.9515 | not reported |
| MMLU-Pro (5-shot, custom-extract) | 0.8113 | not measured | not reported |
| HumanEval pass@1 (EvalPlus) | 0.915 | not measured | 0.896 |
| HumanEval+ pass@1 (EvalPlus) | 0.848 | not measured | 0.860 |
| IFEval prompt-strict | 0.8540 | not measured | 0.8207 |

On raw AIME pass@1, RedHat scores higher than ours (90% vs 83%) — but the gap is **entirely truncation rate** at the 65K max_tokens cap (RedHat truncated 2/30, ours truncated 5/30). On non-truncated pass@1 — the problems where both completed reasoning — all three configs are within 0.4pt of each other (96.0–96.4%). Quantization quality is equivalent across configs on AIME 2024 in this measurement; the differentiator is wall-clock.

## Throughput (4× B300 SXM6, SM 10.3, NVLink, TP=4)

Same hardware, same TP=4, same prompts as the accuracy table above.

| Workload | Operating point | This artifact | RedHat | Ratio |
|---|---|---|---|---|
| AIME 2024 reasoning (thinking=high, c=8) | wall-clock for 30 problems | 476s | 1405s | 2.95× |
| AIME 2024 reasoning | per-request median tok/s | 182.9 | 99.6 | 1.84× |
| Coding (HumanEval chat, c=1) | output tok/s | 278.68 | 131.06 | 2.13× |
| Coding (HumanEval chat, c=4) | output tok/s | 649.35 | 417.87 | 1.55× |
| Coding (HumanEval chat, c=8) | output tok/s | 1104.89 | 673.12 | 1.64× |
| Coding (HumanEval chat, c=16) | output tok/s | 1577.20 | 1007.78 | 1.56× |

Two different ratios to disambiguate:

- **Pure decode throughput**: at c=1 chat coding, ours is 2.13× faster. On AIME reasoning at c=8, the per-request median decode rate is 182.9 vs 99.6 tok/s — a **1.84×** decode speedup.
- **AIME batch wall-clock**: 1405s / 476s = **2.95×**. This includes the truncation-rate differential at the 65K max_tokens cap.

## MTP draft acceptance per workload (B300)

| Workload | Acceptance |
|---|---|
| Random prompts (1024 in / 512 out) | 10.75% |
| Raw code completion (HumanEval `/v1/completions`) | 67.29% |
| Chat-templated code (HumanEval `/v1/chat/completions`, c=1) | 87.96% |
| Chat-templated code, c=4 / c=8 / c=16 | 88.27% / 87.92% / 88.19% |
| Instruction following (IFEval) | ~58.5% |
| AIME 2024 reasoning (thinking=high) | 81.60% |

Acceptance does not degrade under batching — flat at 88.0% ± 0.4% across c=1 to c=16 on chat-templated coding.

## Recommended serve config (B300, TP=4)

```bash
VLLM_TEST_FORCE_FP8_MARLIN=1 \
vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --max-model-len 65536 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
```

`--tensor-parallel-size 4` on B300; **TP=4 is faster than TP=8** on this artifact because the per-rank MoE expert shard is small enough that the additional TP-allreduce overhead beats the extra compute. TP=4 was 6-22% faster than TP=8 at c≥4 in our measurements.

## NVFP4 hardware reality (the bit users keep asking about)

| | B200 / B300 (SM 10.0) | RTX PRO 6000 / DGX Spark (SM 12.0+) |
|---|---|---|
| Disk footprint | NVFP4 packed ✓ | NVFP4 packed ✓ |
| GPU memory footprint | NVFP4 packed ✓ | NVFP4 packed ✓ |
| MoE expert matmul | **Native FP4 tensor cores (tcgen05)** | Marlin BF16 (FP4→BF16 dequant inside kernel) |
| FLOPS / throughput | ~2× vs BF16 | Same as BF16 — no FLOPS win |
| Recommended | **Yes — this artifact** | Use [W4A16 sibling](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) |

Until [CUTLASS#3096](https://github.com/NVIDIA/cutlass/issues/3096) (sm_120a NVFP4 grouped-GEMM garbage) is fixed AND vLLM's NVFP4 oracle is extended to allow sm_120 backends AND `flashinfer_cutedsl_sm12x` learns to apply SwiGLU clamp — three independent upstream tracks — **NVFP4 on consumer Blackwell is storage-only**, not compute. Use W4A16 for those platforms.

## License

MIT, inherited from upstream `deepseek-ai/DeepSeek-V4-Flash`.

## Reproduction

GitHub: <https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp> — calibration scripts, B300 install path, and the open-PR queue tracking upstream contributions extracted from this work.
