# Checkpointing design — Phase 2b gate

**Date:** 2026-05-20
**Status:** Design proposal. Diff to follow once approved.

## Original gate (verbatim)

> Diff for multi-rank save fix + per-layer checkpointing before commit. Show: per-rank subdir → rank-0 merge → unified index logic explicitly, atomic checkpoint writes (.tmp + rename), resume-skip logic, and one measured layer checkpoint size against /scratch headroom.

## Pushback on the gate as literally written

The original gate was scoped before we mapped the `llmcompressor` lifecycle. After investigation:

1. **"Per-layer checkpointing" of weights doesn't help resume calibration.** During calibration, weights stay BF16. What accumulates is per-Linear observer state (min/max/scale stats). `compressor.compress_model(model)` materializes packed FP4/FP8 weights only at save time, *after* calibration completes (`compressed_tensors/compressors/model_compressors/model_compressor.py:138`). A weight-only `.pt` checkpoint at hour 6 of calibration would just be the same BF16 source weights we already have at `/scratch/weights/bf16-mtp/`.

2. **True intra-calibration resume needs upstream `SequentialPipeline` patching.** `SequentialPipeline.__call__` (`llmcompressor/pipelines/sequential/pipeline.py:60`) iterates `subgraphs` and calls `LifecycleCallbacks.sequential_epoch_end(subgraph)` after each one. There's no "start at subgraph N" parameter. To resume, we'd need to snapshot observer state per subgraph, then on restart skip the first N subgraphs' forward passes. That's invasive monkey-patching of a hot path in a library with multiple in-flight PRs (#42209, #41276, #2647 — see `upstream_research_2026_05_20.md`). High risk of breaking on every upstream rebase.

3. **Existing upstream cultural permission:** `vllm-project/llm-compressor#1809` (Sep 2025, open) proposes restructuring the sequential pipeline for parallel calibration. No existing feature request for **intra-calibration resume** specifically. Filing one is the right kind of OSS contribution — exactly the "carry the local hack or upstream the request" fork the standing rules describe.

4. **AWS reservation type:** verified via IMDS today — this instance is **on-demand**, not spot. The original "3-day prepaid spot reservation" handoff phrasing was misleading. Reclaim risk from AWS spot is zero. Residual risk is the normal hardware/process-failure class.

## Three-piece compromise (proposed)

What we DO build, in this repo:

### Piece 1 — atomic merge (.tmp + rename)

**Files touched:** `scripts/merge_rank_subdirs.py`, `scripts/quantize_v4_nvfp4_fp8_mtp.py` (`_CompatModel.save_pretrained`).

**Behavior:** every safetensors write lands at `<file>.tmp` first; once `save_file()` returns, `os.replace(tmp, final)` makes it atomic. `model.safetensors.index.json` writes to `.tmp` then renames. If the process is killed mid-write, the partial `.tmp` is discardable and the previous valid `.safetensors` (or absence) is preserved.

**Why it matters:** the bug that bit us at gate-violation point was *not* spot reclaim — it was a graceful-shutdown race where Phase 2b was killed mid-`save`. Atomic writes mean any partial state on disk is a known-valid intermediate, never a corrupt artifact.

### Piece 2 — resume-skip in `save_pretrained`

**Files touched:** `scripts/quantize_v4_nvfp4_fp8_mtp.py`.

**Behavior:** before writing any shards, check if `<output_dir>/model.safetensors.index.json` already exists AND contains:
- ≥37,000 total keys (full model is ~37,331)
- 256 unique expert IDs (range 0-255)
- ≥6 unconditional MTP keys

If all three hold, the artifact is already complete. Skip the write, log `[save] artifact already complete; skipping save and merge`. The script exits cleanly via the normal completion path.

**Why it matters:** if save crashes mid-way (the from_accelerate AttributeError pattern we suppressed; or a network hiccup during the merge), the operator can re-run the same `torchrun` command. Calibration repeats — that's the cost we accept — but save+merge are idempotent. Combined with the standalone `merge_rank_subdirs.py` recovery tool, partial save state is recoverable without re-running calibration.

### Piece 3 — per-subgraph progress marker

**Files touched:** `scripts/quantize_v4_nvfp4_fp8_mtp.py` (hook `QuantizationModifier.on_event(SEQUENTIAL_EPOCH_END)`).

**Behavior:** monkey-patch `QuantizationModifier.on_event` to ALSO write `<output_dir>/_progress.json` after each subgraph completion. JSON shape:

```json
{
  "started_at_utc": "2026-05-20T19:37:59Z",
  "world_size": 8,
  "samples": 768,
  "total_subgraphs": 45,
  "last_subgraph_completed": 23,
  "last_subgraph_completed_at_utc": "2026-05-20T22:14:03Z",
  "elapsed_seconds": 9384
}
```

Tiny file (~300 bytes). Operator can `cat _progress.json` on a killed run to know: how far did we get, is it worth waiting for a retry, or do we restart fresh.

**Why it matters:** in absence of true resume, the marker turns "did calibration crash or is it still running?" into a one-`cat` answer. Most relevant when an SSH session drops and operator needs to triage.

### What this 3-piece does NOT protect

**Calibration itself.** Pieces 1, 2, 3 all protect save / post-save. A process crash at calibration hour 6 means restarting calibration from subgraph 0. Piece 3 tells the operator how far we got, but the work is gone.

On on-demand B300 in us-west-2a, the failure modes that would lose calibration progress are:
- Hardware failure on the instance (rare; AWS will replace within minutes, but instance state on `/scratch` is gone)
- Process-internal crash from a bug in our code or upstream code (catchable in pre-launch dryrun A/B)
- Operator-initiated kill (e.g. another gate violation)
- Network disconnect followed by mistaken process kill

This residual risk is the cost of not patching the upstream pipeline. We accept it for this artifact and file the upstream feature request to remove it for next time.

## Layer checkpoint size measurement (gate item 4)

- Per-subgraph progress marker: ~300 bytes JSON, written 45 times during calibration → ~14 KB cumulative.
- Per-rank `_rank_NN/` subdir during save: ~70-150 GB depending on which slice (rank 0 carries replicated tensors → 150 GB; ranks 1-3 only experts → 70 GB).
- Final merged artifact: ~538 GB.
- `/scratch` mount: 28 TB total, ~23 TB free as of measurement.

Headroom check: 538 GB artifact + 30% temp overhead = ~700 GB peak during save. /scratch headroom: 23 TB / 700 GB = **~33x margin**. No constraint.

## Upstream contribution (parallel to the 3-piece)

File one new issue against `vllm-project/llm-compressor`: feature request for intra-calibration resume in `SequentialPipeline`. Body should:

- Describe the use case (671B-parameter MoE, ~10-12h calibration on 8x B300).
- Cite #1809 (parallel calibration) as the design space precedent.
- Outline the minimum API surface: serialize observer state per subgraph; skip-already-completed-subgraph parameter on `SequentialPipeline.__call__`.
- Note our compress_module → BF16-preserved-during-calib observation, so the snapshot can be observer-only (small).

Reference our reproducer (the publicly-released `canada-quant` artifact, once it exists) — but **without** brand-dropping in PR body. Maintainers see the contributor identity already.

Also comment on `vllm-project/llm-compressor#2734` (the sibling-filed multi-rank desync issue) with our NVFP4 weight-only confirmation that observer-sync hazard fires even on RTN recipes — that's a separate already-filed bug, this is just adding a data point.

## Implementation order (priority-aligned)

1. Atomic merge (`.tmp + rename`) — Piece 1.
2. Resume-skip — Piece 2.
3. Progress marker — Piece 3.
4. Diff review + commit.
5. 4-rank A/B re-verify with the formal four-number report.
6. Phase 2b relaunch (8-rank, 768 samples, max_seq 512, batch 4).
7. Hour-4 and hour-8 status check-ins (mandatory per the gate).

In parallel (separate tool calls during code dev):
- File vLLM bool() PR — 5 sites in `compressed_tensors.py:621,628,642,650,673`.
- Comment on llm-compressor #2734 with NVFP4 reproducer.
- File llm-compressor intra-calibration-resume feature request.

Deferred until after Phase 2b launches:
- Tier-1 static smoke against the dryrun artifact.
- Tier-2 harness prep (`.env` + `pytest -q tests` dry-run).

## Diff to come

Will follow this design doc with the actual code diff against `scripts/quantize_v4_nvfp4_fp8_mtp.py` and `scripts/merge_rank_subdirs.py` in a single review block before any commit.
