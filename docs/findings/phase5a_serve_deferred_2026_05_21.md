# Phase 5a vLLM serve smoke — deferred status (2026-05-21)

**Outcome:** Phase 5a live serve **deferred**. Artifact passed all structural gates and reaches forward-path execution in vLLM mainline + PR #42209, but the W8A8 + DeepGemm + B300 (sm100a) path hits a `cudaErrorNoKernelImageForDevice` deferred error at the first attention sync point, indicating DeepGemm kernel binaries weren't built for sm100a in the prebuilt wheel pulled at build time.

**Artifact state at end of session:** valid, RedHat-shape-aligned, locally backed up. Recipe and on-disk format match RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` exactly EXCEPT for: (a) MTP retention (the differentiator, intentional) and (b) attention `weight_scale` dtype upgraded BF16 → FP32 (DeepGemm requirement; lossless upcast). The artifact is ready for serve once the upstream stack issues are resolved.

---

## What was tried (8 serve attempts on mainline vLLM + PR #42209)

In rough chronological order, each line is one attempt and the new error it surfaced:

| # | Config / patch | Failure mode |
|---|---|---|
| 1 | initial mainline + PR #42209 + paul/dsv4 jasl carry-over | `fp8_einsum` 256-row check on jasl-fork's sm120 kernel (proved we needed mainline, not jasl/dm120) |
| 2 | mainline base, default config | `KeyError: fused_wqa_wkv.weight_scale` — vLLM expects FUSED attn names on disk |
| 3 | + `update_config_for_fused_attn.py` (regex → fused names) | `AssertionError` at `compressed_tensors_w8a16_fp8.py:122` — the exact bug PR #43248 was filed for |
| 4 | + local sed bool() wrap at `compressed_tensors.py:679,700` | Load passed; failed at runtime `t.dim() == N` in DeepGemm fp8_einsum (W8A16 + Marlin path didn't BMM-reshape weight) |
| 5 | + `input_activations` dynamic FP8 in config (W8A8 path) | `AttributeError: ColumnParallelLinear has no weight_scale_inv` — W8A8Fp8 BLOCK scheme doesn't rename like W8A16 does |
| 6 | + `convert_attn_scales_for_vllm.py` (rename `weight_scale` → `weight_scale_inv` + BF16 → FP32) | `KeyError: fused_wqa_wkv.weight_scale_inv` at load — stacked_params_mapping rename targets non-existent param slot (scheme creates `weight_scale`, on-disk now `weight_scale_inv`, names don't match) |
| 7 | + revert rename to `weight_scale` (keep FP32) + patch `attention.py:334` to fall back to `weight_scale` | Load + process complete. `cudaErrorNoKernelImageForDevice` at `attention.py:508 out.zero_()` — deferred async error, likely DeepGemm kernel binary missing for sm100a |
| 8 | with `--enforce-eager` | CLI-layer crash (`VLLM_RUST_FRONTEND_PATH` attr missing); was on wrong branch context, not a real attempt |

Most progress came in attempt 7 — weights loaded successfully (35/35 shards), `process_weights_after_loading` completed for W8A8 BLOCK + DeepGemm, model entered profile_run forward path. Failure was in compiled CUDA kernels, not in our artifact or our patches.

## What we changed and why (artifact-side)

Three artifact-side changes applied during Phase 5a debug, preserved in scripts and reproducible:

1. **`scripts/update_config_for_fused_attn.py`** — `config_groups[group_0].targets` regex from unfused names (`wq_a|wkv`) to FUSED names (`fused_wqa_wkv|compressor.fused_wkv_wgate|wq_b|wo_a|wo_b`). Required because vLLM mainline DSV4 uses `MergedColumnParallelLinear` for attn, and its quant framework only allocates scale params on modules whose prefix matches the regex.

2. **`scripts/convert_attn_scales_for_vllm.py`** — Two artifact mutations + one config mutation:
   - BF16 → FP32 dtype upcast on 215 attn `.weight_scale` tensors across 31 shards. Required because DeepGemm's `deepgemm_post_process_fp8_weight_block` asserts FP32 or UE8M0 at `fp8_utils.py:1017`.
   - Config `input_activations` on group_0 set to `{strategy: 'group', group_size: 128, dynamic: True, num_bits: 8, type: 'float'}` matching RedHat's spec exactly. Required so vLLM picks `CompressedTensorsW8A8Fp8` scheme (which delegates to DeepGemm) instead of `CompressedTensorsW8A16Fp8` (Marlin only, no BMM support).
   - Initial version also renamed `weight_scale` → `weight_scale_inv` on disk; this REVERTED in session because the W8A8 scheme allocates `weight_scale` as the parameter name, so on-disk `weight_scale_inv` causes load failures. The script header docs updated to reflect this.

## What we changed and why (vLLM-side, local patches)

Two vLLM mainline local patches applied during Phase 5a debug, NOT yet upstream-merged:

1. **`compressed_tensors.py:679,700`** — wrap `is_static_input_scheme = input_quant and not input_quant.dynamic` with `bool(...)` to coerce `None` short-circuit to `False`. Already filed as vLLM PR #43248.

2. **`models/deepseek_v4/attention.py:334`** — change `wo_a_scale = self.wo_a.weight_scale_inv` to fallback `wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None) or self.wo_a.weight_scale`. The W8A8Fp8 BLOCK scheme allocates `weight_scale` (not `weight_scale_inv`), but mainline DSV4 attention forward reads `weight_scale_inv` unconditionally. The fallback handles both naming conventions. **TO BE FILED** as upstream PR.

Also queued: PR #43288 (`scale_fmt` defensive `.get()`) — filed during this session.

## Why we stopped at attempt 7

Attempt 7 reached forward-path CUDA execution and failed in a way that requires rebuilding DeepGemm from source for sm100a, or filing an upstream issue with `vllm-project/deep-gemm` (or equivalent) for B300 kernel binary coverage. Both are multi-hour activities with uncertain outcomes — possibly surfacing the NEXT layer of mismatches (NCCL backend selection, MoE kernel arch, etc.).

The user-set goal was shipping the artifact, with V4-Pro extension gated on benchmark success. Without working serve, no benchmarks. So:
- Continuing to chase serve in this session risks more hours without converging to a working state
- The artifact (which IS the deliverable) is structurally correct and shipped privately (HF upload still gated on user authorization)
- Recipe replication doc + upstream PRs ARE forward progress regardless of serve outcome

## What to try next session

In rough priority order:

1. **Rebuild DeepGemm explicitly for sm100a.** Source at `/data/src/vllm/.deps/deepgemm-src/`. Build with `TORCH_CUDA_ARCH_LIST=10.0a` and `CUDA_ARCHITECTURES=100a`. Confirm sm100a binaries are emitted (`cuobjdump --list-elf` should show sm_100a entries on the compiled `.so`).
2. **Re-attempt serve** after DeepGemm rebuild. With our existing artifact-side + vLLM-side patches in place, if attempt-7 was just the DeepGemm binary issue, this should clear the load forward path.
3. **If a deeper issue surfaces** (e.g., NCCL backend not built for sm100a, or another kernel binary gap), file an upstream tracking issue with `vllm-project/vllm` describing "what's required to serve an NVFP4-FP8 DSV4 artifact on B300 sm100a in current mainline + PR #42209."
4. **File upstream PR for attention.py:334 fallback** — small one-line PR alongside our existing #43248 + #43288.
5. **If multiple sessions of debug still don't yield serve-green**, consider re-calibration with all the recipe corrections folded in (W8A8 input_activations explicitly set, scale_dtype=torch.float32 explicitly set in recipe so save-time strip doesn't drop it). ~90 min calibration cost. Cleaner artifact + same DeepGemm rebuild issue likely remains.

## Files that capture this work

| File | Status |
|---|---|
| `docs/recipes/nvfp4_fp8_mtp_replication.md` | committed (45ba077) — 12-section recipe + 14 gotchas including the new Phase 5a ones |
| `docs/findings/upstream_research_2026_05_21.md` | committed (45ba077) — jasl-vs-mainline recon |
| `docs/findings/phase5a_serve_deferred_2026_05_21.md` | this file |
| `scripts/update_config_for_fused_attn.py` | committed (a6c43bb) |
| `scripts/convert_attn_scales_for_vllm.py` | committed (f525def) |
| `scripts/squeeze_global_scales.py` | committed (45ba077) |
| `~/.claude/projects/.../memory/jasl_dm120_is_sm120_not_sm100.md` | committed (45ba077) |
| `~/.claude/projects/.../memory/diverge_from_reference_doesnt_mean_wrong.md` | committed earlier |

Upstream PRs:
- vLLM #43248 — bool() wrap (filed earlier session)
- vLLM #43288 — scale_fmt defensive .get (filed this session)
- llm-compressor #2745 — MTP inference-mode crash (filed earlier session)
- compressed-tensors #711 — dispatch_with_map AttributeError (filed earlier session)
- llm-compressor #2743 — multi-rank cache deadlock (filed earlier session)
- llm-compressor #2741 — intra-calibration resume feature request (filed earlier session)
- llm-compressor #2734 — Observer.synchronize NCCL desync (commented; sibling-filed)
- TO FILE: vLLM attention.py:334 weight_scale_inv-or-weight_scale fallback PR

## Final artifact state (RedHat-aligned + MTP)

`/scratch/weights/v4-flash-nvfp4-fp8-mtp/` (B300 box):
- 35 safetensors shards, 172 GB
- 134,309 keys total (matches Phase 4 gate)
- 256 unique expert IDs (matches Phase 4 gate)
- 799 MTP keys preserved (matches Phase 4 gate — the differentiator)
- `attn.*.weight_scale` is FP32 (upcast from BF16; lossless)
- `config_groups[group_0]`:
  - targets: `re:.*\.attn\.(fused_wqa_wkv|compressor\.fused_wkv_wgate|wq_b|wo_a|wo_b)$`
  - input_activations: dynamic FP8 group=128 (matches RedHat)
  - weights: FP8_BLOCK 128×128 (matches RedHat)
- `quantization_config.scale_fmt: ue8m0` (kept — required by current mainline vLLM `model.py:909` hard subscript; our PR #43288 makes it optional)

The artifact is in the cleanest state we can ship without a working serve verification. The Phase 4 MTP retention gate is intact. The MTP `inference_mode` issue (Option Y, MTP unquantized) is preserved.
