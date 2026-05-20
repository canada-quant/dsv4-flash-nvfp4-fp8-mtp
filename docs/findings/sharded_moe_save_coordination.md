# Sharded-MoE multi-rank save coordination + `from_accelerate` AttributeError

**Status:** Worked around in this repo by replacing `modify_save_pretrained` with a sharded-MoE-aware wrapper that calls our `save_pretrained` on all ranks (per-rank subdir → rank-0 merge) and skips `to_accelerate`/`from_accelerate`. Upstream PR pending against `neuralmagic/compressed-tensors`.

## Symptom #1 — only one rank's expert slice gets written

Before this fix: 4-rank NVFP4 calibration completes (45/45) subgraphs successfully, but the saved artifact contains only experts 0-63 (rank 0's slice). The other 192 experts are missing from `model.safetensors.index.json`.

## Symptom #2 — `AttributeError: 0 is not an nn.Module`

After every save, the wrapper crashes (on non-rank-0 ranks first, then rank 0 cascades) with:

```
File ".../compressed_tensors/offload/convert/from_accelerate.py", line 62, in from_accelerate
  dispatch_with_map(model, *broadcast_obj)
File ".../compressed_tensors/offload/dispatch.py", line 95, in dispatch_with_map
  module = model.get_submodule(name)
File ".../torch/nn/modules/module.py", line 735, in get_submodule
  raise AttributeError("`" + item + "` is not an nn.Module")
AttributeError: `0` is not an nn.Module
```

## Root cause

Both symptoms trace to `llmcompressor.transformers.compression.compressed_tensors_utils.save_pretrained_wrapper`:

```python
def save_pretrained_wrapper(save_directory, ...):
    compressor = ModelCompressor.from_pretrained_model(model, ...)
    if save_compressed:
        compressor.compress_model(model)         # all ranks
    to_accelerate(model)                          # all ranks
    if is_source_process():                       # RANK 0 ONLY
        original_save_fn(save_directory, **kwargs)
        compressor.update_config(save_directory)
        update_and_save_recipe(...)
        copy_python_files_from_model_cache(...)
    if dist.is_initialized():
        dist.barrier()
    from_accelerate(model)                        # all ranks → crash
```

**Symptom #1:** the `if is_source_process()` gate means only rank 0's `save_pretrained` ever runs. Rank 0's `state_dict()` contains its owned-expert slice (experts 0-63 with this repo's contiguous sharding); the other ranks' state never reaches disk.

**Symptom #2:** `from_accelerate(model)` runs unconditionally on every rank. It broadcasts rank 0's device map (a dict of `name -> device` for every submodule rank 0 sees) and each rank calls `model.get_submodule(name)` for every entry. With expert sharding, rank 1's `model.layers.X.ffn.experts[0]` is `None` (rank 1 doesn't own expert 0), but rank 1 is looking up the name because rank 0 broadcast it. `nn.Module.get_submodule("0")` on a `ModuleList` entry that's `None` raises `AttributeError`.

## Fix in this repo (runtime patch)

`scripts/quantize_v4_nvfp4_fp8_mtp.py` step 1c, applied after `apply_dist_state()`:

```python
def _wrapper(save_directory, quantization_format=None, save_compressed=True, **kwargs):
    compressor = ModelCompressor.from_pretrained_model(model_to_wrap, quantization_format=quantization_format)
    if save_compressed:
        compressor.compress_model(model_to_wrap)
    # ALL ranks save (no is_source_process gate)
    original_save_fn(save_directory, **kwargs)
    if _is_src():
        compressor.update_config(save_directory)
        _update_recipe(model_to_wrap.name_or_path, save_directory)
    if dist.is_initialized():
        dist.barrier()
    # SKIP from_accelerate — crashes on sharded experts. Our
    # save_pretrained handles the merge itself before returning.
```

`_CompatModel.save_pretrained` then implements the actual sharded write:

- `world_size > 1, rank > 0`: filter the state_dict to expert-only keys (`re.search(r"\.experts\.\d+\.", k)`). Replicated tensors (embed, head, attn, MTP, norms) are written ONLY by rank 0 to keep the merged key sets disjoint.
- Each rank writes to `<save_dir>/_rank_NN/`.
- `dist.barrier()` (all ranks finished writing).
- Rank 0 calls `scripts/merge_rank_subdirs.merge_rank_subdirs(save_dir, world_size=ws)` which renames each subdir's shards into globally-numbered top-level names and writes a unified `model.safetensors.index.json`.
- `dist.barrier()` (rank 0 done merging — safe for any rank to return).

The `AttributeError` from `from_accelerate` is also caught in `main()`'s `oneshot()` try/except as non-fatal (the artifact has already been written + merged at that point); this is a backup in case the wrapper replacement misses a code path.

## Verification (4-rank, 16 samples, layer 5 scope)

After fix:
- 4 `_rank_NN/` subdirs written (10259 keys × 30 shards for rank 0; 9024 keys × 26 shards each for ranks 1/2/3).
- Merge unions them into 37,331 keys × 108 shards.
- All 256 expert IDs (range 0-255) present.
- 799 MTP keys present.
- Walltime 10.6 min.

Math check: rank 0's 10259 keys = 9024 (expert keys for 64 experts × ~141 tensors-per-expert-incl-scales) + 1235 (replicated non-expert keys). Ranks 1-3 each: 9024 keys = 64 experts × ~141 tensors. Sum: 10259 + 3×9024 = 37,331. ✓ Disjoint, complete.

## Upstream PR design (queued)

Target: `neuralmagic/compressed-tensors`.

Two distinct fixes — should probably be **two PRs** since the issues are conceptually separate:

**PR-a: `dispatch_with_map` robustness on sharded models.** `compressed_tensors/offload/dispatch.py:95` should `try/except AttributeError` around `model.get_submodule(name)` and skip the offload re-dispatch for modules not present on the current rank (i.e. `None` ModuleList entries). The broadcast device map is rank-0-authored; non-rank-0 ranks legitimately have a subset of those modules. Test case: construct a `nn.ModuleList([Linear(...), None, Linear(...), None])`-style model and assert `dispatch_with_map` doesn't raise.

**PR-b: sharded-MoE save coordination.** `save_pretrained_wrapper` should support a "sharded save" mode where every rank writes its slice to a per-rank subdir and the rank-0 process merges. This is more invasive — needs API design for the merge step, decision on whether `from_accelerate` should be skipped (we say yes for sharded; upstream needs to decide). May be safer to file the design proposal first as a GitHub Discussion, then file the PR after consensus.

PR-a is the low-risk one-liner candidate; ship it first. PR-b can follow as a design-first proposal.
