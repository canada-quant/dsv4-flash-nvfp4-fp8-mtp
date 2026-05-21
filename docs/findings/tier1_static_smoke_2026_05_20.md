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
| `quantization_config.scale_fmt` | **`ue8m0` (injected — REQUIRED by our fork)** | not present (RedHat uses different load path that doesn't read this field) |
| Group 0 (attn) format | `float-quantized` | same |
| Group 1 (experts) format | `nvfp4-pack-quantized` | same |
| Group 0 target regex | `re:.*\.layers\.5\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$` (dryrun-scoped) | `re:.*attn.*(fused_wqa_wkv|wq_b|wo_a|wo_b)$` (production-scoped, fused) |
| Group 1 target regex | `re:.*\.layers\.5\.ffn\.experts\.\d+\.(gate_proj|up_proj|down_proj)$` | `re:.*ffn.*(gate|up|down)_proj$` |

Two divergences from RedHat:

- **Layer scoping** in our targets — artifact of `--dry-run-one-layer`. The production run (no `--dry-run-one-layer` flag) writes unscoped targets matching RedHat's pattern. Not a real divergence.
- **`scale_fmt: ue8m0` injection** — our post-save logic sets this. RedHat's HF config doesn't. **[REVERTED 2026-05-21]** This finding's conclusion was wrong. Our jasl/dm120 vLLM fork's `DeepseekV4Attention.__init__` at `vllm/models/deepseek_v4/nvidia/model.py:904` does a hard subscript `config.quantization_config["scale_fmt"]` — missing key → `KeyError` at worker init → server fails to start. RedHat's artifact loads through a different DSV4 code path (probably stock transformers + main-line vLLM) that doesn't read this key; **our fork does**. The "normalize-to-reference" instinct in this finding led to a Phase 5a load crash and ~15 min recovery. The fix: keep injecting `scale_fmt: ue8m0` in the post-save block. See `memory/diverge_from_reference_doesnt_mean_wrong.md` for the meta-rule. Commit `4ac0a20` was the wrong-direction commit; the post-Phase-5a fix re-adds the injection.

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

## Addendum — Tier 2 result + Tier 3 gating

**Tier 2 (`pytest -q tests/` against the cloned harness at `/data/harness/`):** 195/197 passed in 11.55s. The 2 failures are environment-fixture issues that don't affect actual harness-against-vLLM runs:

- `test_vllm_collect_env_helper_downloads_and_runs_official_script` — tries to download an upstream script over the network; the box doesn't have outbound to wherever it expects.
- `test_runtime_stats_helper_slices_serve_log_per_phase` — looks for a `runtime_stats_summary.json` that requires a real serve-log slice the test fixture isn't providing.

Neither failure indicates a harness bug. The CLI wrappers, test_official_baseline, oracle, lm_eval, toolcall15, and chat-smoke tests all pass — those are the ones that will gate Phase 5 against our artifact.

**Tier 3 (vLLM load smoke) is gated on Phase 2b finishing**, not just on GPU availability. The dryrun artifact's recipe scope was `--dry-run-one-layer` (layer 5 only). Other layers' expert weights are still BF16 in the saved safetensors, but the saved `config.json` declares all `.ffn.experts.\d+.(gate_proj|up_proj|down_proj)` paths as NVFP4. vLLM would reject the type inconsistency at load. To run a meaningful Tier 3 smoke we need the full Phase 2b artifact, plus vLLM PR #42209 (NVFP4 MoE for DSV4) either merged or pulled into `/data/venv-serve`.
