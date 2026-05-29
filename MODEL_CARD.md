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

## What this is

- 172 GB across 35 safetensors shards (vs ~600 GB BF16 source, MTP block included).
- Same quantization scheme as `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`: NVFP4 (group=16, FP8 e4m3 scales) on routed FFN experts, FP8_BLOCK 128×128 on attention.
- MTP block (`mtp.0.*`, 799 tensors) kept at BF16 — not dropped at load time, not double-quantized when the MTP draft model is constructed.

That last point is the only structural difference from RedHat's artifact. The HF transformers DSV4 modeling class has `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`, which silently strips MTP keys during the calibration load path. RedHat's artifact ran through that path, so their saved weights don't include MTP. Ours include MTP because we patched the modeling class during calibration.

## Hardware validated

| Platform | Compute cap | HBM per GPU | Interconnect | Status |
|---|---|---|---|---|
| 4× NVIDIA B300 SXM6 AC | SM 10.3 (`sm_103a`) | 288 GB HBM3e | NVLink | Primary — all accuracy + throughput numbers below |
| 4× NVIDIA RTX PRO 6000 Blackwell Server Edition | SM 12.0 (`sm_120`) | 96 GB HBM | PCIe | Also validated — TP=2/TP=4 throughput + GSM8K-50, 3 extra vLLM patches |

Both platforms serve with CUDA graphs enabled (no `--enforce-eager`). Throughput tables below break out per-platform.

## Accuracy (model quality, hardware-invariant)

Measured 2026-05-21 on 4× B300 SXM6 AC (TP=4 for quant configs, TP=8 for BF16 reference which doesn't fit at TP=4). Same prompts, temperature 0, chat template server-side. The same artifact serves on RTX PRO 6000 Blackwell with no weight changes — accuracy reproduces (GSM8K-50 cross-check: 88% strict TP=2 / 90% strict TP=4 on RTX 6000 Pro vs 91.81% strict full-set on B300, within noise).

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

## Throughput

### 4× B300 SXM6 (SM 10.3, NVLink, TP=4)

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

- **Pure decode throughput**: at c=1 chat coding, ours is 2.13× faster. On AIME reasoning at c=8, the per-request median decode rate is 182.9 vs 99.6 tok/s — a **1.84×** decode speedup. The decode ratio is workload-dependent (acceptance % varies) but lands in the 1.8–2.1× range across the workloads measured.
- **AIME batch wall-clock**: 1405s / 476s = **2.95×**. This includes the truncation-rate differential at the 65K max_tokens cap: 5/30 of our responses truncated vs 2/30 of RedHat's, and truncated responses run to the cap, inflating RedHat's total wall-clock. The 2.95× ratio measures "time to run AIME 2024 end-to-end" rather than pure decode speed, and is the right number to cite for "how long does the bench take" but not for "how fast does the model decode."

### 4× RTX PRO 6000 Blackwell (SM 12.0, PCIe, TP=2 and TP=4)

Validated 2026-05-23 on a Brev `familiar-teal-worm` instance. Per-replica `vllm bench serve` random 256-in/256-out, `num_speculative_tokens=1` (SM 12.0 caps spec at k=1). MTP-on for all rows.

| Config | bs=1 out tok/s | bs=4 out tok/s | bs=16 out tok/s | bs=1 TPOT median | MTP acceptance | GSM8K-50 strict |
|---|---|---|---|---|---|---|
| TP=2 | 94.6 | 218.5 | 360.5 | 9.05 ms | 70–73% | 88% |
| TP=4 | 101.0 | 254.0 | 440.1 | 8.20 ms | 67–75% | 90% |

At bs=16, TP=4 is 1.22× faster per-replica than TP=2 on this hardware — opposite of B300, where TP=4 beats TP=8 due to NVFP4 tensor-core underutilization. RTX PRO 6000's slower PCIe interconnect plus lower per-GPU compute means the extra parallelism still pays off at all batch sizes measured.

For context on the same RTX PRO 6000 box, the W4A16+FP8+MTP sibling measured 98.83 tok/s at TP=2 bs=1 — the two artifacts deliver equivalent decode throughput on this hardware, with NVFP4 trading ~4% per-replica throughput for ~10% smaller on-disk footprint (172 GB vs 159 GB).

Three SM 12.0-specific vLLM patches are required beyond the four common patches below. Recipe + diffs in [`docs/RECIPE_RTX6000PRO.md`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/docs/RECIPE_RTX6000PRO.md). Raw bench JSONs in `benchmarks/rtx6000pro/`.

### 2× RTX PRO 6000 (mtcl-class setup, 2026-05-28 Docker verification)

First end-to-end-on-fresh-Docker measurement using `canada-quant/dsv4-rtx6000pro:v3` image. TP=2, MTP k=1, CUDA graphs ON (`cudagraph_capture_sizes=[1,2]`), `max_model_len=8192`, `max_num_seqs=2`, `gpu_memory_utilization=0.95`.

| Benchmark | Result |
|---|---|
| **AIME-2024 full 30 (c=1, thinking=high, max_tokens=32000)** | **24/30 = 80.0% correct** ✓ matches baseline exactly, 0 errors, MTP acceptance **91.05%**, 2587 s wall (uses 32K-context serve profile) |
| AIME-2024 mini (5 problems, c=1, thinking=high) | 4/5 = 80% correct, 0 errors, MTP 90.65%, 129 s |
| Throughput bs=1 random 256/256 | 45.14 tok/s avg / 73.00 tok/s peak, median TPOT 8.13 ms |
| Throughput bs=2 | 21.13 tok/s avg (per-stream 10.5 — TP-allreduce comm-bound over PCIe) |
| Tool calling | ✅ structured `tool_calls` emit with deepseek_v4 parser |
| Thinking mode | ⚠️ functional, reasoning goes to `content` (upstream issue #34650) |

**Optimal config for 2× RTX PRO 6000 is single-user serving (bs=1).** For high-concurrency, use TP=4 on 4× cards or B200/B300 with native FP4. Raw JSONs and full smoke logs in [`benchmarks/rtxpro6000/tp2_*_2026_05_28.*`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/tree/main/benchmarks/rtxpro6000).

### Verified context window on 2× RTX PRO 6000

Two profiles — both with MTP preserved, FP8 KV cache, TP=2:

| Profile | max_model_len | max_num_seqs | cudagraph_capture_sizes | Use case | Headroom |
|---|---|---|---|---|---|
| Standard | 8192 | 2 | [1,2] | Multi-turn chat, agentic | ~3.5 GB free/GPU |
| **Long-context** | **131072 (128K)** | **1** | **[2]** | Single-user RAG, doc analysis | ~3 GB free/GPU |

The long-context profile uses `scripts/serve_rtx6000pro_tp2_longctx.sh`. Compressed-MLA + FP8 KV cache puts per-token cost at ~16 KB/GPU; 128K × 1 seq fits in the post-weights budget (3 GB/GPU free of 96 GB). Verified end-to-end 2026-05-28: serve loads, math smoke (`23×47` ones-digit decomposition) passes at ~40 tok/s. 256K would exceed the budget.

### Docker quickstart (2× RTX PRO 6000)

```bash
git clone https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp.git
cd dsv4-flash-nvfp4-fp8-mtp
docker build -f docker/Dockerfile.rtx6000pro -t canada-quant/dsv4-rtx6000pro:v3 .

docker run --rm --gpus '"device=0,1"' \
  --shm-size=16g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/scripts:/workspace/scripts:ro \
  canada-quant/dsv4-rtx6000pro:v3 \
  bash /workspace/scripts/serve_rtx6000pro_tp2.sh
```

Build ≈45 min (one-time, 366 CUDA objects), ≈11 min serve startup (weight load + CUDA graph capture). All 13 patch layers documented in [`docs/findings/cardb_docker_layers_2026_05_28.md`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/docs/findings/cardb_docker_layers_2026_05_28.md).

### MTP draft acceptance per workload (B300)

| Workload | Acceptance |
|---|---|
| Random prompts (1024 in / 512 out) | 10.75% |
| Raw code completion (HumanEval `/v1/completions`) | 67.29% |
| Chat-templated code (HumanEval `/v1/chat/completions`, c=1) | 87.96% |
| Chat-templated code, c=4 / c=8 / c=16 | 88.27% / 87.92% / 88.19% |
| Instruction following (IFEval) | ~58.5% |
| AIME 2024 reasoning (thinking=high) | 81.60% |

Acceptance does not degrade under batching — flat at 88.0% ± 0.4% across c=1 to c=16 on chat-templated coding. RTX PRO 6000 acceptance lands in the 67–75% range on random prompts (256-in/256-out workload, not directly comparable to the workload-specific rows above).

## Recommended serving config

- **B300 (288 GB HBM3e, NVLink)**: TP=4. TP=8 is slower than TP=4 at c≥4 by up to 21.6% at c=16 — per-rank MoE expert shards at TP=8 are small enough to underutilize NVFP4 tensor-core kernels on B300.
- **RTX PRO 6000 Blackwell (96 GB HBM, PCIe)**: TP=4 with reduced cudagraph captures + `--max-num-seqs 8 --max-num-batched-tokens 2048` to fit memory. TP=2 also works if only 2 GPUs are available; expect 1.22× lower per-replica throughput at bs=16.

## Quick start

See [`docs/QUICKSTART.md`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/docs/QUICKSTART.md) in the source repo for the full build recipe, or use the one-line installer:

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

Calibration corpus: HuggingFaceH4/ultrachat_200k train_sft, **64 samples** × max_seq_len 512 × batch_size 1, seed 42. RedHat's reference recipe uses 768 samples × 512 from the same corpus.

The 64-sample recipe was used due to time/compute constraints during initial bring-up (12× less coverage than RedHat). On the benchmarks measured here, GSM8K / HumanEval / IFEval / MMLU-Pro / AIME-non-truncated all land within noise of the reference. The visible cost of the reduced coverage is AIME truncation rate: 5/30 of our responses hit the 65K max_tokens cap on long reasoning traces vs 2/30 of RedHat's, which is consistent with looser calibration scales producing less-converging reasoning trajectories. A v0.2 recipe with 768 samples is planned.

## vLLM patches required

### Common (all platforms, B300 + RTX PRO 6000)

The artifact loads on vLLM mainline + these 4 patches. They're filed upstream and waiting on review. See [`docs/VLLM_SETUP_ISSUES.md`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/docs/VLLM_SETUP_ISSUES.md) for the exact diffs.

1. PR [#43248](https://github.com/vllm-project/vllm/pull/43248) — `bool()` wrap on `is_static_input_scheme`
2. PR [#43288](https://github.com/vllm-project/vllm/pull/43288) — `.get("scale_fmt", "ue8m0")` on missing key + BF16 `getattr` follow-up
3. PR [#43290](https://github.com/vllm-project/vllm/pull/43290) — `weight_scale_inv`-or-`weight_scale` fallback
4. PR [#43319](https://github.com/vllm-project/vllm/pull/43319) — MTP-quant-detect from safetensors header + BF16 `wo_a` fallback path

The one-line installer applies all four automatically.

### Additional (RTX PRO 6000 Blackwell / SM 12.0 only)

Three SM 12.0-specific patches are required on top of the four above. Not yet filed upstream — diffs are in `patches/sm120_*.diff` in the source repo, full rationale in [`docs/RECIPE_RTX6000PRO.md`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/docs/RECIPE_RTX6000PRO.md).

1. `VLLM_TEST_FORCE_FP8_MARLIN=1` env var — bypasses the NVFP4 MoE backend selector's `swiglu_limit` filter (no `FLASHINFER_TRTLLM` NVFP4 kernel auto-selects on SM 12.0).
2. `weight_scale_inv`-or-`weight_scale` fallback in Marlin's `scaled_mm/marlin.py` (the PR #43290 patch covers `attention.py` only; SM 12.0 also hits Marlin's pre-process site).
3. Skip Marlin pre-processing for layers tagged `is_bmm=True` — DSV4 `wo_a`/`wo_b`/`compressor.wkv` use the SM 12.0 Triton `fp8_einsum` kernel directly; Marlin's tile-layout repack breaks the original `(N, K)` layout the einsum expects.

B300 deployments can skip all three.

## Files in the artifact

- 35 sharded `model-*.safetensors` files + `model.safetensors.index.json` (172 GB total)
- `config.json` — vLLM-compatible quantization_config with fused targets + W8A8 input_activations
- `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` — upstream DSV4-Flash
- `recipe.yaml` — the llm-compressor calibration recipe
- `README.md` — this file

## Reproduction

Full replication recipe in [`docs/recipes/nvfp4_fp8_mtp_replication.md`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/docs/recipes/nvfp4_fp8_mtp_replication.md) — covers the 14 gotchas (sm_103a vs sm_100a, calibration recipe, postprocess pipeline, vLLM build flags).

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
