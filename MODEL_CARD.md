---
license: other
license_name: deepseek-license
license_link: LICENSE
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

## What this is

- 172 GB across 35 safetensors shards (vs 1.3 TB BF16 source).
- Same quantization scheme as `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`: NVFP4 (group=16, FP8 e4m3 scales) on routed FFN experts, FP8_BLOCK 128×128 on attention.
- MTP block (`mtp.0.*`, 799 tensors) kept at BF16 — not dropped at load time, not double-quantized when the MTP draft model is constructed.

That last point is the only structural difference from RedHat's artifact. The HF transformers DSV4 modeling class has `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`, which silently strips MTP keys during the calibration load path. RedHat's artifact ran through that path, so their saved weights don't include MTP. Ours include MTP because we patched the modeling class during calibration.

## Headline measurements

All numbers measured 2026-05-21 on 4× B300 SXM6 AC (compute_cap 10.3). Quant configs at TP=4, BF16 reference at TP=8 (it doesn't fit at TP=4 on B300). Same prompts, same temperature 0, chat template applied server-side.

| Benchmark | This artifact | BF16 + MTP reference | RedHat (no MTP) |
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

## Wall-clock vs RedHat

Same 4× B300 hardware, same TP=4, same prompts:

| Workload | Operating point | This artifact | RedHat |
|---|---|---|---|
| AIME 2024 reasoning (thinking=high, c=8) | wall-clock for 30 problems | 476s | 1405s |
| Coding (HumanEval chat, c=1) | output tok/s | 278.68 | 131.06 |
| Coding (HumanEval chat, c=4) | output tok/s | 649.35 | 417.87 |
| Coding (HumanEval chat, c=8) | output tok/s | 1104.89 | 673.12 |
| Coding (HumanEval chat, c=16) | output tok/s | 1577.20 | 1007.78 |

The speedup ratio at c=1 chat coding is 2.13×; at AIME reasoning it's 2.95×. Reasoning has longer outputs (~17K tokens vs ~500 for coding), which amplifies MTP's per-step advantage. The speedup is bigger on workloads with long outputs and high token-level predictability.

## MTP draft acceptance per workload

| Workload | Acceptance |
|---|---|
| Random prompts (1024 in / 512 out) | 10.75% |
| Raw code completion (HumanEval `/v1/completions`) | 67.29% |
| Chat-templated code (HumanEval `/v1/chat/completions`, c=1) | 87.96% |
| Chat-templated code, c=4 / c=8 / c=16 | 88.27% / 87.92% / 88.19% |
| Instruction following (IFEval) | ~58.5% |
| AIME 2024 reasoning (thinking=high) | 81.60% |

Acceptance does not degrade under batching — flat at 88.0% ± 0.4% across c=1 to c=16 on chat-templated coding.

## Recommended serving config

TP=4 on 4× B300 (or equivalent Blackwell SXM6 with ≥250 GB HBM each). On this artifact, TP=8 is **slower** than TP=4 at c≥4 batched concurrencies — by up to 21.6% at c=16. Per-rank MoE expert shards at TP=8 are small enough to underutilize NVFP4 tensor-core kernels on B300. TP=4 is the right operating point for this artifact in production.

## Quick start

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) in the source repo for the full build recipe, or use the one-line installer:

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
```

Serving:

```bash
# With MTP spec-decode
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

# Without spec-decode
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8
```

## Differences vs RedHat's NVFP4-FP8 artifact

| Aspect | `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` | This artifact |
|---|---|---|
| NVFP4 on routed FFN experts (group=16) | yes | yes |
| FP8_BLOCK 128×128 on attention | yes | yes |
| MTP `mtp.0.*` weights saved | no — transformers stripped them at load | yes (799 tensors, BF16) |
| Loadable with `--speculative-config method=mtp` | no | yes |
| Coding wall-clock @ c=1 chat | 131 tok/s | 279 tok/s |
| AIME 2024 wall-clock (30 problems, c=8) | 1405s | 476s |

The math of the quantization is the same. The architectural difference is MTP retention.

## Quantization recipe

| Group | Modules | Scheme | Format |
|---|---|---|---|
| attention | `wq_a, wq_b, wkv, wo_a, wo_b` (and fused variants) | FP8_BLOCK 128×128, weight static + input dynamic FP8 group=128 | `float-quantized` |
| experts | `w1, w2, w3` per expert | NVFP4 group=16, weight static + input dynamic="local" FP4 group=16 | `nvfp4-pack-quantized` |
| ignored | `lm_head`, `embed_tokens`, norms, `ffn.gate`, `ffn.shared_experts`, attn `compressor`, attn `indexer`, `attn_sink`, `hc_*` | unquantized (BF16) | n/a |
| MTP block (`mtp.0.*`) | all 799 keys | unquantized (BF16, preserved verbatim) | n/a |

Calibration corpus: HuggingFaceH4/ultrachat_200k train_sft, 64 samples × max_seq_len 512 × batch_size 1, seed 42. RedHat used 768 samples × 512 from the same corpus; the 64-sample recipe is faster and produces calibration scales close enough that quality benchmarks land within noise.

## vLLM patches required

The artifact loads on vLLM mainline + these 5 patches. They're filed upstream and waiting on review. See [`docs/VLLM_SETUP_ISSUES.md`](docs/VLLM_SETUP_ISSUES.md) for the exact diffs.

1. PR [#43248](https://github.com/vllm-project/vllm/pull/43248) — `bool()` wrap on `is_static_input_scheme`
2. PR [#43288](https://github.com/vllm-project/vllm/pull/43288) — `.get("scale_fmt", "ue8m0")` on missing key + BF16 `getattr` follow-up
3. PR [#43290](https://github.com/vllm-project/vllm/pull/43290) — `weight_scale_inv`-or-`weight_scale` fallback
4. PR [#43319](https://github.com/vllm-project/vllm/pull/43319) — MTP-quant-detect from safetensors header + BF16 `wo_a` fallback path

The one-line installer applies all four automatically.

## Files in the artifact

- 35 sharded `model-*.safetensors` files + `model.safetensors.index.json` (172 GB total)
- `config.json` — vLLM-compatible quantization_config with fused targets + W8A8 input_activations
- `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` — upstream DSV4-Flash
- `recipe.yaml` — the llm-compressor calibration recipe
- `README.md` — this file

## Reproduction

Full DSV4 Pro replication template in [`docs/recipes/nvfp4_fp8_mtp_replication.md`](docs/recipes/nvfp4_fp8_mtp_replication.md) — covers the 14 gotchas (sm_103a vs sm_100a, calibration recipe, postprocess pipeline, vLLM build flags).

## Citation

```
@misc{canada-quant-dsv4-flash-nvfp4-fp8-mtp-2026,
  title  = {DeepSeek-V4-Flash NVFP4-FP8 with MTP preserved for vLLM speculative decoding},
  author = {Canada Quant},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP}
}
```

## License

Inherits the upstream DeepSeek-V4-Flash license.

## Acknowledgments

- DeepSeek for V4-Flash and the MTP architecture.
- RedHat AI for the NVFP4-FP8 reference recipe.
- vLLM, llm-compressor, compressed-tensors maintainers.
- PR #42209 contributors (sychen52, xinli-sw, pavanimajety, zyongye) for the DSV4 NVFP4 MoE kernel work that made serving possible.
