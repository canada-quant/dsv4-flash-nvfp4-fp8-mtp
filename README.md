# dsv4-flash-nvfp4-fp8-mtp

Source repo for the DeepSeek-V4-Flash NVFP4-FP8 quantization artifact at [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP).

Same quantization math as `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` — NVFP4 group=16 on routed FFN experts, FP8_BLOCK 128×128 on attention. The only structural difference is that the saved weights include the MTP block (`mtp.0.*`, 799 tensors at BF16), which the HF transformers DSV4 modeling class strips at load time by default. We patched the modeling class during calibration so MTP made it through.

The result is that vLLM can load the artifact with `--speculative-config method=mtp` and use MTP speculative decoding.

## Measurements (4× B300 SXM6 AC, TP=4)

| Benchmark | This artifact | BF16 + MTP reference (TP=8) | RedHat (no MTP, TP=4) |
|---|---|---|---|
| AIME 2024 raw pass@1 (thinking=high, 65K cap) | 25/30 = 83.33% | 25/30 = 83.33% | 27/30 = 90.00% |
| AIME 2024 non-truncated pass@1 | 24/25 = 96.00% | 25/26 = 96.15% | 27/28 = 96.43% |
| AIME wall-clock (30 problems, c=8) | 476s | 490s | 1405s |
| MTP draft acceptance, AIME reasoning | 81.60% | 78.19% | n/a |
| GSM8K strict-match (8-shot) | 0.9181 | 0.9484 / 0.9522 (no-MTP / MTP) | 0.910 (self-reported) |
| MMLU-Pro (5-shot) | 0.8113 | — | — |
| HumanEval EvalPlus pass@1 | 0.915 | — | 0.896 |
| IFEval prompt-strict | 0.8540 | — | 0.8207 |
| Coding tok/s (HumanEval chat, c=1) | 278.68 | n/a | 131.06 |

On AIME 2024 raw pass@1, RedHat scores higher (27/30 vs our 25/30). The gap is entirely truncation rate at the 65K max_tokens cap — both configs solve the problems where they don't run out of tokens at the same rate (96% non-truncated for all three). Quantization quality is equivalent; the differentiator is wall-clock when MTP is enabled.

Full benchmark write-ups in [`docs/benchmarks/`](docs/benchmarks/). Methodology and gotchas in [`docs/findings/`](docs/findings/).

## Quick start

One-line install:

```bash
curl -sL https://raw.githubusercontent.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/main/scripts/install_vllm_with_patches.sh | bash
```

Serve:

```bash
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Full setup in [`docs/QUICKSTART.md`](docs/QUICKSTART.md). The 5 patches + 14 gotchas catalog in [`docs/VLLM_SETUP_ISSUES.md`](docs/VLLM_SETUP_ISSUES.md).

## Why a new repo

The sibling repo `canada-quant/dsv4-flash-w4a16-fp8-mtp` carries the W4A16-GPTQ recipe (different calibration math, different artifact, different hardware target). NVFP4 vs W4A16 is a different code path in vLLM (NVFP4 MoE kernel vs Marlin), different `Modifier` class in llm-compressor (`QuantizationModifier` vs `GPTQModifier`), and lands on a different `Modifier` because the GPTQ Hessian-reduce path hangs on multi-rank B300. Separate repos keep "what does this repo actually produce" easy to answer.

## Repo layout

```
MODEL_CARD.md                    — also serves as HF README
LICENSE                          — MIT (matches upstream DeepSeek-V4-Flash)
docs/
  QUICKSTART.md                  — end-to-end serve recipe
  VLLM_SETUP_ISSUES.md           — the 5 vLLM patches + 14 gotchas
  FINDINGS.md                    — index of findings docs
  benchmarks/                    — per-benchmark write-ups + raw JSONs
  findings/                      — methodology/diagnostic notes
  recipes/                       — calibration replication recipe
patches/
  modeling_deepseek_v4.py.diff   — removes mtp.* from _keys_to_ignore_on_load_unexpected
scripts/
  install_vllm_with_patches.sh   — the one-line installer
  quantize_v4_nvfp4_fp8_mtp.py   — calibration entry point
  postprocess_for_vllm.py        — config + key surgery for vLLM compatibility
  verify_mtp_keys.py             — confirm MTP keys present in saved artifact
  verify_mtp_quantized.py        — confirm MTP weights are NOT quantized (BF16 pass-through)
  aime_bench.py                  — AIME 2024 bench harness with thinking-mode + acceptance capture
vendor/dsv4-upstream/
  model.py, kernel.py, config.json — vendored upstream files (calibration target)
```

## Upstream contributions

Five vLLM patches were extracted from this work and filed upstream:

| PR | Description | Status |
|---|---|---|
| [#43248](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` (compressed-tensors) | open |
| [#43288](https://github.com/vllm-project/vllm/pull/43288) | `.get("scale_fmt", "ue8m0")` + BF16 `getattr` follow-up | open |
| [#43290](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback (DSV4 attention) | open |
| [#43319](https://github.com/vllm-project/vllm/pull/43319) | MTP-quant-detect from safetensors + BF16 `wo_a` fallback path | open |
| [#43297](https://github.com/vllm-project/vllm/issues/43297) | `(1,)`-shape `global_scale` loader broadcast (issue) | open |
| [#43304](https://github.com/vllm-project/vllm/issues/43304) | MTP draft inherits main quant scheme (issue) | partially addressed by #43319 |

Also filed: [llm-compressor #2745](https://github.com/vllm-project/llm-compressor/issues/2745) (MTP inference-mode crash), [compressed-tensors #711](https://github.com/vllm-project/compressed-tensors/issues/711) (sharded-module load path).

## License

MIT, inherited from the upstream DeepSeek-V4-Flash license. See [`LICENSE`](LICENSE).
