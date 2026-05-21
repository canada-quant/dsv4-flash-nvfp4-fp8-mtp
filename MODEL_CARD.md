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

**The first DeepSeek-V4-Flash NVFP4-FP8 quantization that preserves the MTP speculative-decoding layer.**

- 📦 Artifact: 172 GB across 35 safetensors shards (vs 1.3 TB BF16 source — **~25% memory footprint**)
- 🎯 Quantization math: matches RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` recipe
- 🆕 Differentiator: `mtp.0.*` weights present + serve-loadable with `--speculative-config method=mtp`
- 🏆 Quality: **matches BF16 reference** on AIME 2024 (83.3% / 83.3%) at half the GPU budget
- ⚡ Throughput: **2.95× faster than RedHat** on long-output reasoning workloads (AIME, thinking=high)

## Headline results — measured 2026-05-21 on 4× B300 SXM6 AC

| Benchmark | This artifact (TP=4) | BF16 + MTP reference (TP=8) | RedHat no-MTP (TP=4) |
|---|---|---|---|
| **AIME 2024 raw pass@1** | 83.33% | **83.33%** (parity) | 90.00% |
| **AIME 2024 non-truncated pass@1** | 96.00% | 96.15% | 96.43% |
| AIME wall-clock (30 problems, c=8) | **476s** | 490s | 1405s |
| **MTP acceptance — AIME reasoning** | **81.60%** | 78.19% | n/a (no MTP) |
| GSM8K strict-match (8-shot) | 0.9181 | 0.9484 (no-MTP) / 0.9522 (MTP) | 0.910 (self-reported) |
| GSM8K flexible-extract (8-shot) | 0.9515 | 0.9477 / 0.9515 | — |
| MMLU-Pro (5-shot) | 0.8113 | — | — |
| HumanEval EvalPlus pass@1 | 0.915 | — | 0.896 |
| HumanEval+ pass@1 | 0.848 | — | 0.860 |
| IFEval prompt-strict | 0.8540 | — | 0.8207 |

The cross-quant headline: **on raw AIME pass@1 RedHat scores higher (90% vs 83%), but the gap is entirely truncation-rate** (RedHat truncated 2/30 at 65K cap, we truncated 5/30). On the **non-truncated pass@1 — problems where both completed reasoning — all three configs are equivalent (96.0–96.4%)**. The quantization recipe preserves reasoning quality; the speedup story is the differentiator.

## The MTP differentiator — measured speedup vs RedHat

Wall-clock decode speedup on production workloads, same hardware, same prompts, temperature 0, chat-template applied server-side:

| Workload | Operating point | Ours (NVFP4+MTP, TP=4) | RedHat (NVFP4, no MTP, TP=4) | Speedup |
|---|---|---|---|---|
| **AIME 2024 reasoning (thinking=high)** | c=8 wall-clock | **476s** | 1405s | **2.95×** |
| Coding (HumanEval chat, c=1) | output tok/s | **278.68** | 131.06 | **2.13×** |
| Coding (HumanEval chat, c=4) | output tok/s | 649.35 | 417.87 | 1.55× |
| Coding (HumanEval chat, c=8) | output tok/s | 1104.89 | 673.12 | 1.64× |
| Coding (HumanEval chat, c=16) | output tok/s | 1577.20 | 1007.78 | 1.56× |

**Speedup is highest on long-output workloads (reasoning)** because MTP's per-step advantage compounds with output length, and reasoning content has high token-level structure that the MTP draft head predicts well.

## MTP draft-acceptance rate per workload

The MTP draft head's acceptance rate is content-dependent:

| Workload | Acceptance | Notes |
|---|---|---|
| Random prompts (1024 in / 512 out) | 10.75% | Worst case — high per-token entropy |
| Raw code completion (HumanEval `/v1/completions`) | 67.29% | Pure code prediction |
| Chat-templated code (HumanEval `/v1/chat/completions`) | **87.96–88.27%** (flat c=1→c=16) | Code + prose wrapper |
| Instruction following (IFEval) | ~58.5% | Mixed format |
| **AIME 2024 reasoning (thinking=high)** | **81.60%** | Math + LaTeX + step structure |

**Acceptance does NOT degrade under batching** — measured flat at 87.92–88.27% across c=1, 4, 8, 16 on chat-templated coding.

## Recommended serving config

**TP=4 on 4× B300 (or equivalent Blackwell SXM6 with ≥250GB HBM each).** Counterintuitively, TP=8 is **slower** than TP=4 on this artifact at batched concurrencies (c≥4) by up to 21.6%, because per-rank MoE expert shards become small enough to underutilize NVFP4 tensor-core kernels. TP=4 is the right operating point for production.

Verified on 4× B300 SXM6 AC, compute_cap 10.3 (`sm_103a`).

## Quick start (vLLM)

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) in the source repo for the full build recipe (`TORCH_CUDA_ARCH_LIST=10.3a`, `CUDA_HOME=/usr/local/cuda`, the 5 local patches needed until upstream merges).

```bash
# Quick serve recipe (after vLLM is built with the patches — see QUICKSTART.md)

# With MTP spec-decode (the differentiator):
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

# Plain serve (no spec-decode):
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8
```

## How this differs from RedHat's NVFP4-FP8

| Aspect | RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8 | This artifact |
|---|---|---|
| NVFP4 on routed FFN experts (group=16, FP8 e4m3 scales) | yes | yes |
| FP8_BLOCK 128×128 on attention | yes | yes |
| MTP `mtp.0.*` weights | **dropped at load** (transformers `_keys_to_ignore_on_load_unexpected`) | **preserved** (our `patches/modeling_deepseek_v4.py.diff`) |
| MTP weights serve-loadable | n/a (no MTP weights to load) | **yes** — verified end-to-end on vLLM mainline + 5 local patches |
| Spec-decode via `--speculative-config method=mtp` | not possible | works |
| Wall-clock on AIME (vs each other) | 1405s | **476s (2.95× faster)** |

## Quantization recipe

| Group | Modules | Scheme | Format |
|---|---|---|---|
| group_0 (attention) | `wq_a, wq_b, wkv, wo_a, wo_b` and fused variants | FP8_BLOCK 128×128, weight static + input dynamic FP8 group=128 | `float-quantized` |
| group_1 (routed FFN experts) | `w1, w2, w3` per expert | NVFP4 group=16, weight static + input dynamic="local" FP4 group=16 | `nvfp4-pack-quantized` |
| ignored | `lm_head`, `embed_tokens`, norms, `ffn.gate`, `ffn.shared_experts`, attn `compressor`, attn `indexer`, `attn_sink`, `hc_*` | unquantized (BF16) | n/a |
| MTP block (`mtp.0.*`) | all 799 keys | **unquantized BF16** (preserved verbatim) | n/a |

Calibration corpus: HuggingFaceH4/ultrachat_200k train_sft, 64 samples × max_seq_len 512 × batch_size 1, seed 42.

## Files in the artifact

- 35 sharded `model-*.safetensors` files + `model.safetensors.index.json` (172 GB total)
- `config.json` — vLLM-compatible quantization_config with fused targets + W8A8 input_activations
- `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` — upstream DSV4-Flash
- `README.md` — this file

## vLLM patches required (until upstream merges)

The artifact loads on vLLM mainline + these patches. See [`docs/VLLM_SETUP_ISSUES.md`](docs/VLLM_SETUP_ISSUES.md) in the source repo:

1. **PR #43248** — `bool()` wrap on `is_static_input_scheme` (compressed_tensors)
2. **PR #43288** — `.get("scale_fmt", "ue8m0")` on missing key (DSV4 model.py)
3. **PR #43288 follow-up** — `getattr(config, "quantization_config", None) or {}` for BF16 model loading (no quant_config attr)
4. **PR #43290** — `weight_scale_inv`-or-`weight_scale` fallback (DSV4 attention.py)
5. **PR #43319** — MTP-quant-detect from safetensors header + BF16 `wo_a` fallback path (so MTP block isn't double-quantized when already BF16 on disk)

## Reproduction recipe

Full DSV4 Pro replication template in [`docs/recipes/nvfp4_fp8_mtp_replication.md`](docs/recipes/nvfp4_fp8_mtp_replication.md) — covers the 14 gotchas (sm_103a vs sm_100a, calibration recipe, postprocess pipeline, vLLM build flags, etc.).

## Citing this work

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

- DeepSeek for V4-Flash and the MTP architecture
- RedHat AI for the NVFP4-FP8 reference recipe
- vLLM, llm-compressor, compressed-tensors maintainers
- DSV4 NVFP4 MoE PR #42209 (sychen52, xinli-sw, pavanimajety, zyongye) for the kernel work that made serve possible
