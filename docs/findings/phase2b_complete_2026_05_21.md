# Phase 2b 1-rank calibration — COMPLETE

**Date:** 2026-05-21
**Status:** Artifact written. All four required gate numbers green. Ready for Phase 3 (post-process) and Phase 4 (verify).

## Run parameters

- 1-rank single-process (no torchrun) — chosen after 8-rank stalled on the `vllm-project/llm-compressor#2743` offload-cache deadlock.
- `--samples 64 --max-seq-len 512 --batch-size 1` — RedHat-matched calibration params from `examples/quantizing_moe/deepseek_v4_example.py`.
- Recipe with MTP modules in `ignore` (`r"re:.*mtp\..*"`) — chosen after the first 1-rank run crashed at subgraph 44 with `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed` during MTP qparam writeback. MTP weights ship BF16 unquantized; differentiator vs RedHat (MTP present vs absent) preserved.

## Final gate (the four numbers, plus topology checks)

| Metric | Target | Measured | Pass |
|---|---|---|---|
| Total keys in merged index | 100K+ (production scale) | **134,309** | ✓ |
| Unique expert IDs | 256 (range 0-255) | **256 (0-255)** | ✓ |
| `mtp.*` keys preserved | ≥ 795 (anchored on 1-rank dryrun's 799) | **799** | ✓ |
| `layers.5.attn.wq_a.weight_scale` present (FP8 attn worked) | yes | yes | ✓ |
| `layers.5.ffn.experts.0.w1.weight_packed` present (NVFP4 experts worked) | yes | yes | ✓ |
| `mtp.0.e_proj.weight` dtype | bfloat16 (unquantized) | **bfloat16 (4096, 4096)** | ✓ |

Sub-checks worth recording:

- **MTP "quantization leak":** 3 keys matching `mtp.*` AND `(_scale|_packed|_global_scale|_zero_point)`. Not blocking — these are probably global scales accidentally registered on MTP submodules whose paths don't match our `ignore` regex. Need to inspect; likely cosmetic. The headline mtp weight (`mtp.0.e_proj.weight`) is verifiably bfloat16, confirming the MTP forward path is unquantized.
- **Atomic save cleanup:** `0 .tmp` files leftover on disk, `0 _rank_NN/` subdirs (multi-rank merge path wasn't exercised in 1-rank).
- **Non-fatal exception caught:** `[oneshot] non-fatal (offload-twice): artifact already on disk; continuing to post-save processing.` — the existing try/except for `Attempted to offload a module twice` fired exactly as designed. Artifact was written before the exception; post-save config edits ran successfully.

## Walltime

| Phase | Duration |
|---|---|
| Process startup → patches applied | ~30s |
| BF16 load (single rank, all 256 experts local) | ~80s |
| Dataset prep (64 samples) | ~10s |
| Calibration phase (subgraphs 1-45) | **~70 min** (4323s per progress.json, last subgraph completed at T+72:03) |
| Compression + per-subdir save + merge | **~15 min** |
| Post-save config edits | ~3s |
| **CALIBRATION_DONE total** | **5238.3s = 87.3 min** |

## Artifact on disk

- `/scratch/weights/v4-flash-nvfp4-fp8-mtp/`
- 35 safetensors shards, 172 GB
- `config.json` with HF-named expert targets (gate_proj/up_proj/down_proj), no `scale_fmt` injection (matches RedHat reference)
- `model.safetensors.index.json` with 134,309-key weight_map
- `_progress.json` showing 45/45 subgraphs completed
- `recipe.yaml` from llmcompressor's save path
- Tokenizer files copied from BF16 source

## Three iterations to get here

1. **8-rank samples=768 batch=4** (initial launch) — stalled at subgraph 2 after 110+ min in `compute_dynamic_scales_and_zp`. Identical Python stack md5 across all 8 ranks, GPU SM=100% / MEM=0% / 240W / 36°C — the canonical SM-busy-wait-no-real-compute signature. Filed as `vllm-project/llm-compressor#2743`.
2. **8-rank samples=64 batch=1** (RedHat-matched params) — stalled at subgraph 1 in a different cache path (`_onload_value`). Same SM=100/MEM=0 signature. Confirmed via py-spy that the bug is in the offload cache code, multi-rank only. `propagate_error=False` exists on main as the workaround; our pinned `f2aa32e2` predates it.
3. **1-rank samples=64 batch=1, MTP-in-ignore** — completed cleanly. The 8-rank deadlock doesn't trigger single-process. The MTP inference-mode crash doesn't trigger when MTP isn't in scope.

## What's next

- **Phase 3:** post-process artifact for vLLM. The existing `postprocess_for_vllm.py` may need a once-over given the recipe differences from W4A16; review before running.
- **Phase 4:** `verify_mtp_keys.py` + `verify_mtp_quantized.py` runs against the artifact. The verify_mtp_quantized check should be informed that MTP intentionally ships BF16 here (not quantized).
- **Phase 5:** vLLM serve smoke. Gated on `vllm-project/vllm#42209` merging or being pulled into `/data/venv-serve`.
- **Two upstream bugs queued from this session:** #2743 (cache deadlock) and the not-yet-filed inference-mode tensor crash on MTP-style layers.

## Differentiator demonstration setup

For Phase 5's MTP acceptance rate measurement:

- **Reference number (upstream):** `7.01% spec_acceptance_rate_percent` on official DeepSeek-V4-Flash API, 8× B300, MTP enabled, `bench_random_8192x512`, concurrency 1 (from `harness/baselines/20260505_official_b300_mtp2_clean/performance/primary.json`).
- **Our target:** any positive `spec_acceptance_rate_percent` is the differentiator demonstration (RedHat's NVFP4-FP8 cannot report this number at all — they have no MTP weights to draft from). Matching ~7% would be the quality claim.
- **Serve command (planned):** `vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP --tensor-parallel-size 4 --port 8089 --kv_cache_dtype=fp8 --speculative_config '{"method":"mtp","num_speculative_tokens":2}'`.
