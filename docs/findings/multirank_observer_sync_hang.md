# Multi-rank `Observer.synchronize` desync hang with expert-sharded MoE

**Status:** Reproduced 2026-05-20 on B300 SM 10.0a, 4 ranks. Monkey-patched in this repo at `scripts/quantize_v4_nvfp4_fp8_mtp.py` (`world_size > 1` gate, before `oneshot()` is called). Upstream PR pending against `neuralmagic/llm-compressor` — tag `kylesayrs`.

## Symptom

4-rank NVFP4 calibration via `llmcompressor.oneshot(..., recipe=QuantizationModifier(...))` hangs **indefinitely** at subgraph 6 of 45 in `SequentialPipeline`. GPUs report 100% utilization (queued kernels not draining). No NCCL timeout fires because the collective never completes — it's a permanent desync, not a slow operation.

dmesg shows a single `NVRM: knvlinkSendInbandData_IMPL: Failed to send inband data: 0` line at process startup time only — NOT the continuous cascade pattern of the GPTQ-path B300 NVLink bug. This is software desync, not hardware.

## py-spy diagnosis (the load-bearing artifact)

```
$ sudo /data/venv-calib/bin/py-spy dump --pid <worker_pid>
Thread <pid> (active): "MainThread"
    __torch_function__ (torch/utils/_device.py:122)
    update_offload (compressed_tensors/offload/cache/cpu.py:48)
    decorate_context (torch/utils/_contextlib.py:124)
    __setitem__ (compressed_tensors/offload/cache/base.py:213)
    update_offload_parameter (compressed_tensors/offload/__init__.py:145)
    sync_activation_observers (llmcompressor/modifiers/quantization/quantization/mixin.py:296)
    on_event (llmcompressor/modifiers/quantization/quantization/base.py:104)
    update_event (llmcompressor/modifiers/modifier.py:122)
    event (llmcompressor/core/lifecycle.py:204)
    sequential_epoch_end (llmcompressor/core/session_functions.py:165)
    __call__ (llmcompressor/pipelines/sequential/pipeline.py:163)
    ...
```

Same stack on every rank, same frame across 3+ samples 3 seconds apart — frozen, not slow.

## Root cause (one level deeper than the first read suggested)

`QuantizationModifier`'s main file has zero `dist.*` calls — this was verified earlier and was the basis for "NVFP4 multi-rank should work" in the original pivot rationale. **That claim was correct at the file level and wrong at the call-graph level.**

`mixin.sync_activation_observers` (called from `QuantizationModifier.on_event(SEQUENTIAL_EPOCH_END, ...)`) iterates over matched modules and calls `observer.synchronize()` on each one. Two observer base classes do per-module async all-reduce:

`llmcompressor/observers/base.py:138`:
```python
def synchronize(self) -> List[dist.Work]:
    comms = []
    for attr, op in [
        ("past_min_vals",        dist.ReduceOp.MIN),
        ("past_max_vals",        dist.ReduceOp.MAX),
        ("past_global_min_vals", dist.ReduceOp.MIN),
        ("past_global_max_vals", dist.ReduceOp.MAX),
    ]:
        val = getattr(self, attr, None)
        if val is not None:
            comms.append(
                dist.all_reduce(as_broadcastable(val), op=op, async_op=True)
            )
    return comms
```

`llmcompressor/observers/moving_base.py:102`:
```python
def synchronize(self) -> List[dist.Work]:
    comms = []
    world_size = dist.get_world_size()
    for attr in ("past_min_vals", "past_max_vals", "past_global_min_vals", "past_global_max_vals"):
        val = getattr(self, attr, None)
        if val is not None:
            val.div_(world_size)
            comms.append(
                dist.all_reduce(as_broadcastable(val), op=dist.ReduceOp.AVG, async_op=True)
            )
    return comms
```

The returned `pending_comms` list is then awaited by `wait_for_comms(pending_comms)` in `mixin.sync_activation_observers`.

With **expert-sharded MoE**, each rank's `MoE.__init__` places `Expert(...)` modules only for its owned slice (e.g. rank 0 owns experts 0-63, rank 1 owns 64-127, etc.); non-owned slots are `None`. `match_named_modules(model, resolved_targets, ignore)` therefore returns a **different set of modules per rank**. Each rank iterates its own (disjoint) set and calls `dist.all_reduce` on observers that exist only on its side. **NCCL all-reduce requires every participating rank to call it the same number of times in the same order**; with disjoint sets, ranks call all-reduce on different module identities — the collective never matches up. Permanent desync.

## Why the original pivot rationale missed this

The pivot from GPTQ to NVFP4 was motivated by the verified absence of `dist.*` in `llmcompressor/modifiers/quantization/` (the `QuantizationModifier`'s own file). The check was too narrow — it should have followed the call graph one hop deeper into `mixin.sync_activation_observers` and from there into `observer.synchronize`. The lesson is durable: "X has no Y calls" claims need to follow the function calls one level, not stop at the file boundary. Recorded as a meta-rule for the next session.

## Fix (in-repo, runtime monkey-patch)

`scripts/quantize_v4_nvfp4_fp8_mtp.py` step 1b, applied after `apply_dist_state()`:

```python
if world_size > 1:
    import llmcompressor.observers.base as _obs_base
    import llmcompressor.observers.moving_base as _obs_moving
    _obs_base.Observer.synchronize = lambda self: []
    _obs_moving.MovingAverageObserverBase.synchronize = lambda self: []
```

Returning an empty list makes `pending_comms` empty, `wait_for_comms([])` returns immediately, and each rank uses its own local observer statistics for the subsequent `recompute_qparams()` call.

## Why skipping cross-rank sync is *correct* for this recipe (not a hack)

NVFP4 + FP8_BLOCK as used here are **RTN-style**: scales are derived from the weight tensors themselves, not from observer activation statistics. The recipe's `QuantizationScheme` does not opt into dynamic input quantization with cross-rank sample averaging — the observers exist but their stats are not consulted by the scale-computation path.

A/B verification on layer 5, expert 0, w1/w2/w3 `weight_scale`:

| | 1-rank value | 4-rank-with-patch value | ratio |
|---|---|---|---|
| w1 |mean| | 1.778e+02 | 1.778e+02 | 1.000 |
| w2 |mean| | 1.751e+02 | 1.751e+02 | 1.000 |
| w3 |mean| | 1.794e+02 | 1.794e+02 | 1.000 |

Attention `weight_scale` on layer 5 (wq_a/wq_b/wkv/wo_a/wo_b) match in shape, dtype, |min|, |max|, mean **exactly** between 1-rank and 4-rank-with-patch.

For activation-quantized recipes (W4A8, dynamic-input W8A8) the patch *would* lose cross-rank sample averaging — only ~96 calibration samples per rank instead of 768 fully averaged. For min/max observers that's still plenty, but for percentile/MSE observers it could shift the scale. This is the design space for the upstream PR.

## Upstream PR design (queued, not yet filed)

Target: `neuralmagic/llm-compressor`, tagging `kylesayrs`.

Proposed fix is **Design B** from the brand-strategy notes — explicit `expert_sharded` flag on the observer config, conditionally skips per-observer `dist.all_reduce` when the user has opted in. Simpler than auto-detection via `all_gather` of "do you own this module?", lower review surface, won't surprise users running standard DDP.

PR body should include:
- This py-spy diagnostic as the reproducer.
- The A/B numbers showing scales are scheme-derived not observer-derived for RTN recipes.
- A unit test that constructs a sharded-MoE model (single rank, simulating the disjoint match) and asserts `synchronize()` doesn't hang at `wait_for_comms`.
- Link to `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` model card as the real-world reproducer.

vLLM #41511 (W4A16 MoE TP sharding) is the prior contact with kylesayrs and should be mentioned for context.
