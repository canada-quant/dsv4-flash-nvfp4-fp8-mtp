# Tier-1 static smoke against the 4-rank A/B re-verify artifact (2026-05-20)

**Status:** PASS with one fixable config-label finding.

**Artifact under test:** `/scratch/weights/v4-flash-nvfp4-dryrun-4rank` (the dryrun used to clear the Phase 2b gate; 4-rank, layer 5 scope, 16 samples). This is structurally identical to what Phase 2b will produce except for the recipe's scope (production is all layers, not just layer 5).

## What was checked

1. **`config.json` structure** — `architectures`, `model_type`, `num_hidden_layers`, `expert_dtype`, `num_nextn_predict_layers`, `quantization_config.format`, `quantization_config.config_groups`.
2. **`model.safetensors.index.json` integrity** — total key count, unique expert IDs, mtp.* key count.
3. **Scale dtype audit** — sampled tensor dtypes at the quantized layer 5 attn + experts.
4. **vLLM `CompressedTensorsConfig.from_config()` parse** — does vLLM's quantization config parser accept our saved config?

## Results

### `config.json` (matches RedHat reference shape)

| Field | Our artifact | RedHat `NVFP4-FP8` reference |
|---|---|---|
| `architectures` | `["DeepseekV4ForCausalLM"]` | same |
| `model_type` | `deepseek_v4` | same |
| `expert_dtype` | `fp4` | same |
| `num_nextn_predict_layers` | `1` (MTP slot present) | `1` |
| `quantization_config.format` | `mixed-precision` | same |
| `quantization_config.scale_fmt` | **`ue8m0` (injected)** | **NOT PRESENT** |
| Group 0 (attn) format | `float-quantized` | same |
| Group 1 (experts) format | `nvfp4-pack-quantized` | same |
| Group 0 target regex | `re:.*\.layers\.5\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$` (dryrun-scoped) | `re:.*attn.*(fused_wqa_wkv|wq_b|wo_a|wo_b)$` (production-scoped, fused) |
| Group 1 target regex | `re:.*\.layers\.5\.ffn\.experts\.\d+\.(gate_proj|up_proj|down_proj)$` | `re:.*ffn.*(gate|up|down)_proj$` |

Two divergences from RedHat:

- **Layer scoping** in our targets — artifact of `--dry-run-one-layer`. The production run (no `--dry-run-one-layer` flag) writes unscoped targets matching RedHat's pattern. Not a real divergence.
- **`scale_fmt: ue8m0` injection** — our post-save logic (carried from sibling W4A16 path) sets this field. RedHat does not. The per-group `scale_dtype` (`torch.float8_e4m3fn`) already specifies the FP8-scale format for the experts group; a top-level `scale_fmt` is redundant and risks being misinterpreted by future vLLM versions. **Action:** removed the injection from `scripts/quantize_v4_nvfp4_fp8_mtp.py` post-save block. Phase 2b is already running with the old script — the final artifact will get a one-liner post-clean after Phase 2b finishes.

### Index integrity (matches gate report)

- 37,331 total keys
- 256 unique expert IDs (0-255)
- 799 mtp.* keys

### Scale dtype audit (correct format)

Layer-5 quantized attn (FP8_BLOCK):
- `weight`: `float8_e4m3fn` ✓
- `weight_scale`: `bfloat16` (8x32 for wq_a — block-grid shape, FP16 scales are correct for FP8_BLOCK)

Layer-5 quantized experts (NVFP4):
- `weight_packed`: `uint8` (2048×2048, 2 nibbles per byte) ✓ — correct NVFP4 packing
- `weight_scale`: `float8_e4m3fn` ✓ — matches RedHat's `scale_dtype: torch.float8_e4m3fn`
- `weight_global_scale`: `float32` (shape `(1,)`) ✓
- `input_global_scale`: `float32` (shape `(1,)`) ✓ — present, confirming the recipe is activation-quantized (matches RedHat's `input_activations` settings)

Initial confusion in this smoke about a "missing `.weight` key" for NVFP4 experts was a smoke-script bug: NVFP4 packing uses `.weight_packed`, not `.weight`. Fixed in the smoke output above.

### vLLM CompressedTensorsConfig parse

`from compressed_tensors.compressed_tensors import CompressedTensorsConfig; CompressedTensorsConfig.from_config(qc)` — succeeds, returns parsed config with:

- `quant_format = "mixed-precision"`
- `ignore` = 8771 entries (all the non-quantized layer paths from the dryrun)
- `target_scheme_map` with both group regexes registered

No exception, no warning about unrecognized fields.

## Open questions deferred to tier-3 (vLLM load smoke, post-Phase-2b)

- Does vLLM actually pick `compressed_tensors_moe_w4a4_nvfp4` as the MoE backend at load time, or does it fall back to a different scheme?
- Does the bfloat16 FP8_BLOCK `weight_scale` need explicit `scale_dtype` annotation in the attn group's `weights` block? RedHat's reference doesn't set it explicitly either — vLLM appears to infer it from the tensor dtype on disk.
- Does the MTP layer load via `--speculative_config method=mtp num_speculative_tokens=2`? This is THE differentiator for the artifact and requires a live GPU + the merged vLLM PR #42209 to test.

## Decision

Tier 1 verdict: **PASS**. No blocker for Phase 2b's production launch (already running). One config-label cleanup (`scale_fmt` removal) committed for the future-state. Tier 2 (harness prep) is next.
