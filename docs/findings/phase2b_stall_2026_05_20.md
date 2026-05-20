# Phase 2b 8-rank stall — all ranks wedged at `compute_dynamic_scales_and_zp`

**Date:** 2026-05-20
**Status:** Diagnosed and worked around (RedHat-matched params + `propagate_error=False`). Diagnostic data preserved at `/data/nvfp4-mtp/logs/phase2b_stall_diag_20260520T230231Z/` on the B300 box.

## Symptom

Phase 2b launched 21:12:29 UTC with `--samples 768 --max-seq-len 512 --batch-size 4 --nproc-per-node 8`. Subgraph 1 completed in 316s. Subgraph 2 then ran for **110+ minutes with zero progress** before manual intervention:

- `_progress.json` stuck at `last_subgraph_completed: 1`.
- Log file (`/data/nvfp4-mtp/logs/phase2b_8rank.log`) mtime stuck at 21:18 UTC — no writes for 100+ min.
- All 8 GPUs: SM utilization 100%, **memory utilization 0%**, power draw ~240 W (B300 peak is ~700 W), temperature 35-39 °C (idle range; full load would be 60-80 °C).
- All 8 worker py-spy stacks: **byte-for-byte identical md5** (`a1f24276ca8f4032d7a298ab2c8f9126`), wedged at:
  ```
  __torch_function__ (torch/utils/_device.py:122)
  __torch_function__ (torch/utils/_device.py:122)
  calculate_range (compressed_tensors/quantization/utils/helpers.py:215)
  calculate_qparams (compressed_tensors/quantization/utils/helpers.py:75)
  compute_dynamic_scales_and_zp (compressed_tensors/quantization/utils/helpers.py:196)
  forward_quantize (compressed_tensors/quantization/lifecycle/forward.py:314)
  quantized_forward (compressed_tensors/quantization/lifecycle/forward.py:276)
  forward (compressed_tensors/offload/module.py:58)
  ```

This is the canonical signature of a multi-rank deadlock — either NCCL collective desync or a CUDA-stream interlock — masked by an SM busy-wait kernel that holds 100% util while doing no actual memory traffic.

## Root cause hypothesis (working theory)

`compute_dynamic_scales_and_zp` is the path that computes per-token-group input quantization scales for NVFP4 expert inputs (`dynamic: "local"` in the recipe's `input_activations` config). When the calibration data dispatched to the 8 ranks produces routing-imbalanced expert activations — some ranks see more tokens for a given expert than others — the per-Linear forward count diverges across ranks within a single subgraph step. Anything inside `compute_dynamic_scales_and_zp` that synchronizes (implicit or explicit) then desyncs.

This is a separate hazard from the `Observer.synchronize` desync we already worked around (issue #2734). That one was about explicit `dist.all_reduce` per matched module. This one appears to be about an **implicit interlock inside the dynamic-input-quant path** that fires per-forward — and at our `samples=768, batch_size=4, seq=512` the call count is high enough to surface the race that `samples=64, batch_size=1` doesn't.

## Workaround (this repo)

Two changes in `scripts/quantize_v4_nvfp4_fp8_mtp.py`:

1. **Match RedHat's reference calibration parameters exactly.** Per `vllm-project/llm-compressor:examples/quantizing_moe/deepseek_v4_example.py` on branch `deepseekv4-experimental` (the script that produced `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` with GSM8K 0.910):
   - `NUM_CALIBRATION_SAMPLES = 64` (our default was 768 from the W4A16 predecessor recipe — wrong)
   - `MAX_SEQUENCE_LENGTH = 512` (matches)
   - `batch_size = 1` (our default was 4)
   - `shuffle_calibration_samples = True`

2. **Set `propagate_error=False` in `oneshot()`.** RedHat's reference script does this with the comment "work around reliance on transformers cache." Without it, the 8-rank stall reproduces deterministically at subgraph 2 with our sample/batch settings. We don't fully understand the mechanism — `propagate_error` is an llmcompressor pipeline flag whose internals aren't documented — but RedHat shipped a working artifact with it, so it's known-safe.

## Why the 4-rank dryrun didn't catch this

The 4-rank dryrun used `--samples 16 --batch-size 4 --dry-run-one-layer` — only layer 5 was quantized, and the calibration sample count was tiny. The per-forward call count into `compute_dynamic_scales_and_zp` was orders of magnitude lower, so the race condition that fires at scale didn't surface. The Tier-2 A/B gate verified mathematical correctness (scales identical), not multi-rank scalability at production volume.

**Lesson:** the gate's four numbers (total keys / unique experts / MTP keys / scale ratio) are necessary but not sufficient. A "production-scope" stall test — 1 layer of recipe but FULL `--samples` and `--batch-size` — would have surfaced this in ~10 min instead of after 1h45m of stalled Phase 2b. Adding this to the gate design for future sessions.

## Diagnostic data preserved

`/data/nvfp4-mtp/logs/phase2b_stall_diag_20260520T230231Z/` on the B300 box contains:

- `pyspy_pid_*.txt` × 8 (one per worker rank, all identical md5)
- `pyspy_master_*.txt` (torchrun master)
- `nvidia_smi.csv` (SM=100%, MEM=0%, power=240W, temp=36°C on all 8 GPUs)
- `dmesg_knvlink.txt` (empty — no NVLink hardware errors during the stall)

`/data/nvfp4-mtp/logs/phase2b_stalled_compute_dynamic_scales_*.log` is the full stalled-run log (200 KB; last write 21:18 UTC).

## Upstream report candidate

Worth filing against `vllm-project/llm-compressor` once we've confirmed the workaround produces a clean Phase 2b run. The body:

- Title: `[Bug] Multi-rank calibration with NVFP4 + dynamic="local" input activations deadlocks at compute_dynamic_scales_and_zp without propagate_error=False`
- Reproducer: 8-rank torchrun, 256-routed-expert MoE, NVFP4 experts with `input_activations.dynamic: "local"`, `samples >= ~256, batch_size >= 4`, `propagate_error` left at default (whatever that is — needs to be checked in llmcompressor source).
- Workaround: `propagate_error=False` matches the in-tree DSV4 example and avoids the stall.
- Ask: document `propagate_error` semantics, or default it to False for sequential-pipeline MoE recipes.

## Relaunch plan

`torchrun --nproc-per-node 8 ... --samples 64 --max-seq-len 512 --batch-size 1` (new defaults). Expected walltime ~3-6h based on subgraph 1's 316s × 45 subgraphs scaled for the 12× lower sample count: roughly 316s × (64/768) × 45 = ~1200s **but** that's the wrong math because subgraph 1 included setup overhead. Realistic projection: **2-4h** for the full 45 subgraphs at the new params.
