# dsv4-flash-nvfp4-fp8-mtp

Reproduction repo for [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) — NVFP4 routed experts + FP8 block 128×128 attention + **BF16 Multi-Token Prediction (MTP) draft head retained** on DeepSeek-V4-Flash. Same quantization math as [`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8), but the MTP block (`mtp.0.*`, 799 tensors) is preserved at BF16 so vLLM can load it with `--speculative-config method=mtp`.

Full model card with TL;DR, benchmarks, throughput, and honest limitations lives on the [HF page](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP); this README is the operator/reproduction tour.

## Family / related repos

| Repo | HF model card | Role |
|---|---|---|
| **this repo** (`dsv4-flash-nvfp4-fp8-mtp`) | [NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) | NVFP4 routed experts + MTP, Blackwell-native (B300 / RTX PRO 6000) |
| [`canada-quant/dsv4-flash-w4a16-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp) | [W4A16-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) | sibling — W4A16 routed experts (Hopper-compatible), same MTP-retention pattern. **SM 12.0 / RTX PRO 6000 deployments doing batched thinking-mode**: use this NVFP4 repo instead — the W4A16 Marlin MoE decode path corrupts ~50% of long thinking generations under concurrent load. See [debug log](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp/blob/main/docs/findings/sm12x_token_corruption_2026_05_24.md) and [`jasl/vllm#12`](https://github.com/jasl/vllm/issues/12). |
| [`canada-quant/dsv4-flash-w4a16-fp8`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8) | [W4A16-FP8](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) | predecessor (no-MTP baseline) — broadest hardware compatibility |
| [`canada-quant/dsv4-pro-nvfp4-fp8-mtp`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp) | [Pro NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) | larger sibling — V4-Pro NVFP4 + MTP, B300-only |

## Headline measurements

### 4× B300 SXM6 AC (Blackwell SM 10.3, sm_103a), TP=4

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

### 4× RTX PRO 6000 Blackwell (SM 12.0, sm_120), TP=2 and TP=4

Validated 2026-05-23 on Brev. MTP-on for all rows, k=1 (SM 12.0 cap).

| Config | bs=1 output tok/s | bs=4 output tok/s | bs=16 output tok/s | bs=1 TPOT median | MTP acceptance | GSM8K-50 strict |
|---|---|---|---|---|---|---|
| TP=2 | 94.6 | 218.5 | 360.5 | 9.05 ms | 70–73% | 88% |
| TP=4 | **101.0** | 254.0 | **440.1** | **8.20 ms** | 67–75% | 90% |

#### AIME-2024 deep thinking-mode concurrency sweep (2026-05-25)

TP=4, cuda graphs ON (capture sizes [1,2,4,8]), MTP `num_speculative_tokens=1`, `max-model-len=16384`. Bench JSONs at [`benchmarks/rtxpro6000/cardb_aime30_c{1,2,4,8}_thinking.json`](benchmarks/rtxpro6000/).

| Concurrency | Correct/30 | Stop / Length | Errors | Wall (s) | Problems/min | MTP accept | Speedup vs c=1 |
|---|---|---|---|---|---|---|---|
| c=1 (sequential) | **24/30** (80.0%) | 22 / 8 | 0 | 1453.9 | 1.24 | 90.61% | 1.0x |
| c=2 | **23/30** (76.7%) | 23 / 7 | 0 | 787.6 | 2.29 | 90.75% | 1.85x |
| c=4 | **21/30** (70.0%) | 20 / 10 | 0 | 386.6 | 4.66 | 90.93% | 3.76x |
| c=8 | (terminated) | n/a | n/a | n/a | n/a | n/a | — |

**Key findings:**
- **0 errors and 0 stopped-but-wrong across c=1/2/4** — every wrong answer is length-truncated at max_tokens (not a quality issue, a budget issue). Non-truncated pass@1 is essentially 100% at all clean concurrencies.
- **MTP acceptance stable at 90.6–90.9%** regardless of concurrency. The NVFP4 `flashinfer_trtllm` MoE backend is rock-solid on SM 12.0 under all tested batch sizes.
- **c=8 throughput collapse**: TP=4 with no NVLink (PCIe-only) caused combined throughput to drop from 450 t/s @ c=4 to ~38 t/s @ c=8 — a 12× per-request slowdown. MTP itself stayed healthy; the bottleneck is TP-allreduce communication over PCIe at high concurrency. **Recommendation for higher aggregate throughput on RTX PRO 6000: run 2 replicas at TP=2 instead of 1 replica at TP=4 c=8.**

Full benchmark write-ups in [`docs/benchmarks/`](docs/benchmarks/). Methodology and gotchas in [`docs/findings/`](docs/findings/).

## Quick start

One-line installer (applies all common patches):

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
```

Serve with MTP spec-decode (B300):

```bash
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

For RTX PRO 6000 (SM 12.0), see [`docs/RECIPE_RTX6000PRO.md`](docs/RECIPE_RTX6000PRO.md) — needs 3 additional patches and the `VLLM_TEST_FORCE_FP8_MARLIN=1` env var. Full setup at [`docs/QUICKSTART.md`](docs/QUICKSTART.md). 5 patches + 14 gotchas catalog at [`docs/VLLM_SETUP_ISSUES.md`](docs/VLLM_SETUP_ISSUES.md).

## vLLM patch series — [`vllm-patches/`](vllm-patches/)

The full minimum patch series we apply on top of `jasl/vllm@a02a3778f` to serve this artifact on RTX PRO 6000 (SM 12.0). Each patch is documented with its upstream PR link, status, and rationale in [`vllm-patches/README.md`](vllm-patches/README.md):

| Patch | Purpose | Upstream |
|---|---|---|
| `0001_marlin_moe_archs_40923.patch` | Build native sm_120a Marlin MoE cubins (eliminates JIT-PTX corruption) | [PR #40923](https://github.com/vllm-project/vllm/pull/40923) (open) |
| `0002_marlin_moe_workspace_4x.patch` | Oversize Marlin MoE lock-array workspace 4× (defensive) | (to file as follow-up to #40923) |
| `0003_marlin_moe_c_tmp_36889.patch` | Drop `min()` clamp on `c_tmp` FP32 reduce buffer (block decode safety) | [PR #36889](https://github.com/vllm-project/vllm/pull/36889) (closed, re-file candidate) |

Card B works cleanly with patch 0001 alone (NVFP4 routed experts use `flashinfer_trtllm` MoE backend, which doesn't hit the Marlin moe wna16 paths that 0002/0003 fix). The patches matter for the W4A16-MTP sibling — applying all three on the same build is the recommended setup for parity.

## Why a new repo (vs the W4A16 sibling)

NVFP4 vs W4A16 is a different code path in vLLM (NVFP4 MoE kernel vs Marlin), a different `Modifier` class in llm-compressor (`QuantizationModifier` vs `GPTQModifier`), and lands on a different `Modifier` because the GPTQ Hessian-reduce path hangs on multi-rank B300. Separate repos keep "what does this repo actually produce" easy to answer.

## Repo layout

```
MODEL_CARD.md                    — mirror of the HF README
LICENSE                          — MIT (matches upstream DeepSeek-V4-Flash)
docs/
  QUICKSTART.md                  — end-to-end serve recipe
  VLLM_SETUP_ISSUES.md           — the 5 vLLM patches + 14 gotchas
  RECIPE_RTX6000PRO.md           — RTX PRO 6000 (SM 12.0) specific recipe
  FINDINGS.md                    — index of findings docs
vllm-patches/                    — minimum patch series for current jasl/vllm
  README.md                      — per-patch status, upstream PRs, rationale
  0001_marlin_moe_archs_40923.patch
  0002_marlin_moe_workspace_4x.patch
  0003_marlin_moe_c_tmp_36889.patch
  benchmarks/                    — per-benchmark write-ups + raw JSONs
  findings/                      — methodology/diagnostic notes
  recipes/                       — calibration replication recipe
patches/
  modeling_deepseek_v4.py.diff   — removes mtp.* from _keys_to_ignore_on_load_unexpected
  sm120_*.diff                   — RTX PRO 6000 (SM 12.0) additional patches
scripts/
  install_vllm_with_patches.sh   — one-line installer
  quantize_v4_nvfp4_fp8_mtp.py   — calibration entry point
  postprocess_for_vllm.py        — config + key surgery for vLLM compatibility
  verify_mtp_keys.py             — confirm MTP keys present
  verify_mtp_quantized.py        — confirm MTP weights are NOT quantized (BF16 pass-through)
  aime_bench.py                  — AIME 2024 bench harness with thinking-mode + acceptance capture
benchmarks/rtx6000pro/           — RTX PRO 6000 raw bench JSONs (2026-05-23)
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
