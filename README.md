# dsv4-flash-nvfp4-fp8-mtp

Reproduction repo for [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) — NVFP4 routed experts + FP8 block 128×128 attention + **BF16 Multi-Token Prediction (MTP) draft head retained** on DeepSeek-V4-Flash. Same quantization math as [`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8), but the MTP block (`mtp.0.*`, 799 tensors) is preserved at BF16 so vLLM can load it with `--speculative-config method=mtp`.

> ## 🛑 Hardware scope: B200 / B300 only
>
> This artifact targets **datacenter Blackwell (SM 10.0 / sm_100a / sm_103a)** — the only hardware where the native NVFP4 tensor-core path (`tcgen05`) actually executes the MoE matmul in FP4. On consumer Blackwell (RTX PRO 6000 SE / DGX Spark / RTX 5090, SM 12.0+), vLLM must fall back to the Marlin BF16 kernel which dequantizes FP4 → BF16 inside the kernel. That fallback works correctly but **forfeits the FP4 FLOPS advantage**, and our 2026-05-28 measurements showed throughput parity with — and footprint disadvantage vs — the W4A16 sibling on the same hardware.
>
> **If you're deploying on RTX PRO 6000, DGX Spark, RTX 5090, or any consumer Blackwell, use the W4A16 sibling instead:**
>
> - GitHub: [`canada-quant/dsv4-flash-w4a16-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp)
> - HF: [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP)
> - Smaller on-disk (~159 GB vs 172 GB), slightly higher throughput on Marlin INT4 (its native scheme), full published bench coverage on RTX PRO 6000.

## Family / related repos

| Repo | HF model card | Role |
|---|---|---|
| **this repo** (`dsv4-flash-nvfp4-fp8-mtp`) | [NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) | **B200 / B300 only.** NVFP4 routed experts + MTP, datacenter Blackwell native FP4 tcgen05 path |
| [`canada-quant/dsv4-flash-w4a16-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp) | [W4A16-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) | **Recommended for RTX PRO 6000 / DGX Spark / consumer Blackwell + H200.** W4A16 routed experts, broad hardware support, same MTP-retention pattern |
| [`canada-quant/dsv4-flash-w4a16-fp8`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8) | [W4A16-FP8](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) | predecessor (no-MTP baseline) — broadest hardware compatibility |
| [`canada-quant/dsv4-pro-nvfp4-fp8-mtp`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp) | [Pro NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) | larger sibling — V4-Pro NVFP4 + MTP, B300-only |

## Headline measurements — 4× B300 SXM6 AC (Blackwell SM 10.3, sm_103a), TP=4

| Benchmark | This artifact | BF16 + MTP reference (TP=8) | RedHat NVFP4 (no MTP, TP=4) |
|---|---|---|---|
| AIME 2024 raw pass@1 (thinking=high, 65K cap) | 25/30 = 83.33% | 25/30 = 83.33% | 27/30 = 90.00% |
| AIME 2024 non-truncated pass@1 | 24/25 = 96.00% | 25/26 = 96.15% | 27/28 = 96.43% |
| AIME wall-clock (30 problems, bs=8) | **476 s** | 490 s | 1405 s |
| MTP draft acceptance, AIME reasoning | 81.60% | 78.19% | n/a |
| GSM8K strict-match (8-shot) | 0.9181 | 0.9484 / 0.9522 (no-MTP / MTP) | 0.910 (self-reported) |
| MMLU-Pro (5-shot) | 0.8113 | — | — |
| HumanEval EvalPlus pass@1 | 0.915 | — | 0.896 |
| IFEval prompt-strict | 0.8540 | — | 0.8207 |
| Coding output tok/s (HumanEval chat, bs=1) | **278.68** | n/a | 131.06 |

On AIME 2024 raw pass@1, RedHat scores higher (27/30 vs 25/30) — the gap is **entirely truncation rate** at the 65K max_tokens cap (96% non-truncated for all three). Quantization quality is equivalent; the differentiator is wall-clock when MTP is enabled.

## Quick start — B300 / B200 server (datacenter Blackwell)

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash

CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

B300 path uses mainline `vllm-project/vllm@main` + 5 patches (PR #43248, #43288, #43290, #43319, #42209) — see `scripts/install_vllm_with_patches.sh` for the one-shot installer.

## NVFP4 hardware reality

| | B200 / B300 (SM 10.0) | RTX PRO 6000 / DGX Spark (SM 12.0+) |
|---|---|---|
| Disk footprint | NVFP4 (~half FP8) ✓ | NVFP4 packed ✓ |
| GPU memory footprint | NVFP4 packed ✓ | NVFP4 packed ✓ |
| MoE expert matmul | **Native FP4 tensor cores (tcgen05)** | Marlin BF16 (FP4→BF16 dequant inside kernel) |
| FLOPS / throughput | ~2× vs BF16 | Same as BF16 — no FLOPS win |
| Recommended | **Yes** | Use W4A16 sibling instead |

Three things would need to land before native FP4 works on consumer Blackwell:

1. **CUTLASS sm_120a NVFP4 grouped-GEMM fix** ([CUTLASS#3096](https://github.com/NVIDIA/cutlass/issues/3096)) — TMA-WS tactics currently produce garbage; needs CUDA 13.0 + `compute_120f`.
2. **vLLM NVFP4 oracle** ([source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py)) currently gates server-Blackwell backends on `device_capability_family(100)`. Open PRs: [#41738](https://github.com/vllm-project/vllm/pull/41738), [#43333](https://github.com/vllm-project/vllm/pull/43333), [#43341](https://github.com/vllm-project/vllm/pull/43341), [#43687](https://github.com/vllm-project/vllm/pull/43687).
3. **SwiGLU clamp support in `flashinfer_cutedsl_sm12x`** — the only sm_120-compatible NVFP4 MoE kernel doesn't apply the SwiGLU clamp that DeepSeek-V4 sets at `swiglu_limit=10.0`. Currently unclaimed upstream.

Until all three land, **W4A16 is the practical choice on consumer Blackwell**.

## Why a new repo (vs the W4A16 sibling)

NVFP4 vs W4A16 is a different code path in vLLM (NVFP4 MoE kernel vs Marlin), a different `Modifier` class in llm-compressor (`QuantizationModifier` vs `GPTQModifier`), and lands on a different `Modifier` because the GPTQ Hessian-reduce path hangs on multi-rank B300. Separate repos keep "what does this repo actually produce" easy to answer.

## Repo layout

```
MODEL_CARD.md                    — mirror of the HF README
LICENSE                          — MIT (matches upstream DeepSeek-V4-Flash)
docs/
  QUICKSTART.md                  — end-to-end serve recipe (B300)
  VLLM_SETUP_ISSUES.md           — the 5 vLLM patches + 14 gotchas
  FINDINGS.md                    — index of findings docs
scripts/
  install_vllm_with_patches.sh   — one-line B300 installer
  quantize_v4_nvfp4_fp8_mtp.py   — calibration entry point
  postprocess_for_vllm.py        — config + key surgery for vLLM compatibility
  verify_mtp_keys.py             — confirm MTP keys present
  verify_mtp_quantized.py        — confirm MTP weights are NOT quantized (BF16 pass-through)
patches/
  modeling_deepseek_v4.py.diff   — removes mtp.* from _keys_to_ignore_on_load_unexpected
vendor/dsv4-upstream/
  model.py, kernel.py, config.json — vendored upstream files (calibration target)
```

## Upstream contributions filed during this work

Five vLLM patches extracted from this work and filed upstream:

| PR / Issue | Description | Status |
|---|---|---|
| [`vllm-project/vllm#43248`](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` (compressed-tensors) | open |
| [`vllm-project/vllm#43288`](https://github.com/vllm-project/vllm/pull/43288) | `.get("scale_fmt", "ue8m0")` + BF16 `getattr` follow-up | open |
| [`vllm-project/vllm#43290`](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback (DSV4 attention) | open |
| [`vllm-project/vllm#43319`](https://github.com/vllm-project/vllm/pull/43319) | MTP-quant-detect from safetensors + BF16 `wo_a` fallback path | open |
| [`vllm-project/vllm#43297`](https://github.com/vllm-project/vllm/issues/43297) | `(1,)`-shape `global_scale` loader broadcast (issue) | open |
| [`vllm-project/vllm#43304`](https://github.com/vllm-project/vllm/issues/43304) | MTP draft inherits main quant scheme (issue) | partially addressed by #43319 |

Also filed: [`vllm-project/llm-compressor#2745`](https://github.com/vllm-project/llm-compressor/issues/2745) (MTP inference-mode crash), [`vllm-project/compressed-tensors#711`](https://github.com/vllm-project/compressed-tensors/issues/711) (sharded-module load path).

PR [`vllm-project/vllm#42209`](https://github.com/vllm-project/vllm/pull/42209) (sychen52 et al., NVIDIA) which added the DSV4 NVFP4 MoE kernel merged 2026-05-22; this artifact serves on top of that.

## License

MIT, inherited from upstream `deepseek-ai/DeepSeek-V4-Flash`. See [`LICENSE`](LICENSE).
