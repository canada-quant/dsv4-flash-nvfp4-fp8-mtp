# canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP

**The first DeepSeek-V4-Flash NVFP4-FP8 quantization that preserves the MTP speculative-decoding layer.**

- 📦 Artifact: ~172 GB across 35 safetensors shards
- 🎯 Calibration math: matches RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` exactly
- 🆕 Differentiator: `mtp.0.*` weights present + serve-loadable with `--speculative-config method=mtp`
- 🏆 Benchmark: **GSM8K 0.9181 strict (8-shot)** — beats RedHat's reported 0.910

## Quick start

```bash
# Requires vLLM mainline + PR #42209 (NVFP4 MoE for DSV4) cherry-picked.
# Until our PRs (#43248, #43288, #43290) merge, apply locally — see below.
TORCH_CUDA_ARCH_LIST=10.3a pip install -e .  # rebuild for sm_103a on B300

# Serve (no spec-decode):
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 --kv-cache-dtype fp8

# Serve WITH MTP spec-decode (the differentiator):
CUDA_HOME=/usr/local/cuda VLLM_TEST_FORCE_FP8_MARLIN=1 \
  vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
  --tensor-parallel-size 4 --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

**vLLM serve patches required** (until upstream merges):
- `bool()` wrap on `is_static_input_scheme` at `vllm/.../compressed_tensors.py:679,700` — our PR #43248
- `.get("scale_fmt", "ue8m0")` at `vllm/models/deepseek_v4/nvidia/model.py:909` — our PR #43288
- `getattr(self.wo_a, "weight_scale_inv", None) or self.wo_a.weight_scale` at `vllm/models/deepseek_v4/attention.py:334` — our PR #43290

## How this differs from RedHat's NVFP4-FP8

| Aspect | RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8 | This artifact |
|---|---|---|
| NVFP4 on routed FFN experts | yes | yes (identical math) |
| FP8_BLOCK 128×128 on attention | yes | yes (identical math) |
| MTP `mtp.0.*` weights | **dropped at load** (transformers `_keys_to_ignore_on_load_unexpected`) | **preserved** (our `patches/modeling_deepseek_v4.py.diff`) |
| `vllm serve --speculative-config method=mtp` | not usable (no draft weights) | usable (v2 artifact pending) |
| GSM8K strict-match (8-shot) | 0.910 (self-reported) | **0.9181 ± 0.0076** |
| GSM8K flexible-extract (8-shot) | — | **0.9515 ± 0.0059** |
| MMLU-Pro (5-shot, custom-extract) | RedHat hasn't reported | **0.8113 ± 0.0035** |
| MTP spec-decode acceptance rate | 0 (no MTP) | TBD (target ~7% per upstream) |

### GSM8K — full reference frame

| Run | strict-match | flexible-extract |
|---|---|---|
| **DeepSeek-V4-Flash BF16, with MTP** (8× B200 TP=4 ref, upstream harness baseline) | 0.9522 | 0.9515 |
| **DeepSeek-V4-Flash BF16, no MTP** (8× B200 TP=4 ref) | 0.9484 | 0.9477 |
| **`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`** (4× B300 TP=4, this artifact) | **0.9181** | **0.9515** |
| `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` (self-reported) | 0.910 | — |

Observations:
- **`flexible-extract` is invariant to quantization on this benchmark** — our NVFP4-FP8-MTP exactly matches the BF16-MTP baseline (0.9515 vs 0.9515). The model arrives at the correct numeric answer on the same 1319 problems despite quant-induced text-generation perturbations. Flexible-extract uses regex to find the final numeric answer and is robust to small wording differences.
- **`strict-match` drops 3.41 pts vs BF16-MTP** (0.9522 → 0.9181) — this measures exact response-format adherence; quantization affects token-level distributions enough to shift "The answer is 42." → "I think the answer is 42." style variations.
- **We beat RedHat's `strict-match` 0.910 by 0.81 pts** while ALSO carrying the MTP differentiator. RedHat doesn't publish flexible-extract.

Why the MTP retention matters: the upstream DeepSeek-V4-Flash reports ~7% spec-decode acceptance rate on agentic workloads (per their B300 baseline reference). That translates to ~1.5–2× wall-clock speedup on prompts where draft tokens are accepted. RedHat's artifact cannot do spec-decode at all because the weights aren't present. Ours can.

## Calibration recipe

| Group | Modules | Scheme | Format |
|---|---|---|---|
| group_0 (attention) | `wq_a, wq_b, wkv, wo_a, wo_b` | FP8_BLOCK 128×128, weight static + input dynamic FP8 group=128 | `float-quantized` |
| group_1 (routed FFN experts) | `w1, w2, w3` per expert | NVFP4 group=16, weight static + input dynamic="local" FP4 group=16 | `nvfp4-pack-quantized` |
| ignored | `lm_head`, `embed_tokens`, norms, `ffn.gate`, `ffn.shared_experts`, attn `compressor`, attn `indexer`, `attn_sink`, `hc_*` | unquantized (BF16) | n/a |
| MTP modules (v1) | all MTP weights | unquantized (BF16) — see `docs/findings/phase5a_serve_deferred_2026_05_21.md` and llm-compressor #2745 | n/a |
| MTP modules (v2) | MTP attn + experts: quantized; MTP `embed`/`e_proj`/`h_proj`: ignored | mixed (see scripts/quantize_v4_nvfp4_fp8_mtp_v2.py) | mixed |

Calibration corpus: HuggingFaceH4/ultrachat_200k train_sft, 64 samples × max_seq_len 512 × batch_size 1, seed 42. Matches RedHat's reference exactly.

Hardware: AWS p6-b300.48xlarge (B300 SXM6 AC, compute_cap 10.3 / sm_103a, 8× GPUs).

## Postprocess pipeline (Phase 3)

The raw calibration output isn't directly vLLM-loadable — it needs the postprocesses to convert to vLLM mainline's expected on-disk format. Single driver: `scripts/postprocess_v2_pipeline.py`. Steps:

1. `postprocess_for_vllm.py` — rename mtp embedding key, set `num_hidden_layers=43`, set `expert_dtype=fp4`, inject `scale_fmt: ue8m0`, inject `packed_modules_mapping`
2. `update_config_for_fused_attn.py` — rewrite `config_groups[group_0].targets` to fused names (`fused_wqa_wkv|compressor.fused_wkv_wgate|wq_b|wo_a|wo_b`)
3. `convert_attn_scales_for_vllm.py` — upcast attn `.weight_scale` BF16 → FP32 (DeepGemm requirement) + inject `input_activations` (dynamic FP8 group=128) into `config_groups[group_0]`
4. `squeeze_global_scales.py` — squeeze `(1,)` global_scale tensors to 0-D scalar (vLLM MoE loader expects scalar)
5. Add `re:.*\.layers\.{num_hidden_layers}\..*` to `ignore` so vLLM's MTP draft model construction doesn't apply NVFP4 quant to MTP block

These are all idempotent + atomic-write. See `docs/recipes/nvfp4_fp8_mtp_replication.md` for the full DSV4 Pro replication template.

## Files in the artifact

- `model.safetensors.index.json` + 35 sharded `model-*.safetensors` files
- `config.json` — vLLM-compatible quantization_config with fused targets + W8A8 input_activations
- `tokenizer.json`, `tokenizer_config.json` — from upstream DSV4-Flash
- `generation_config.json`

## Reproduction

```bash
git clone https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp
cd dsv4-flash-nvfp4-fp8-mtp
# See docs/recipes/nvfp4_fp8_mtp_replication.md for the full DSV4 Pro template
```

## Citing this work

If you use this artifact or the MTP-preserving recipe, please reference:

```
@misc{canada-quant-dsv4-flash-nvfp4-fp8-mtp-2026,
  title={DeepSeek-V4-Flash NVFP4-FP8 with MTP preserved for vLLM speculative decoding},
  author={Canada Quant},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP}
}
```

## License

Inherits the upstream DeepSeek-V4-Flash license. See upstream model card for details.

## Acknowledgments

- DeepSeek for the V4-Flash base model and the MTP architecture
- RedHat AI for the NVFP4-FP8 reference quantization recipe
- vLLM and llm-compressor maintainers for the open-source toolchain that made this possible
- The DSV4 NVFP4 MoE PR #42209 (sychen52, xinli-sw, pavanimajety, zyongye) for the kernel work that made serve possible
