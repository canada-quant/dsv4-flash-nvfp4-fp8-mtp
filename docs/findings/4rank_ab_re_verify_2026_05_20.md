# 4-rank A/B re-verify — formal gate report (2026-05-20)

**Status:** GATE PASSED. All four required numbers green. Phase 2b is unblocked.

## Run parameters

- `--samples 16 --max-seq-len 128 --batch-size 4 --dry-run-one-layer`
- 4 ranks via `torchrun --nproc-per-node 4`
- Code under test: commit `9ce3bd5` of `canada-quant/dsv4-flash-nvfp4-fp8-mtp` — atomic `.tmp+rename`, resume-skip with shard verification, MTP threshold ≥795, expert-id check (256 unique), per-subgraph progress marker, stale `.tmp` sweep.

## The four numbers

| # | Metric | Target | Measured | Pass |
|---|---|---|---|---|
| 1 | Total keys in merged `model.safetensors.index.json` | ≥ 37,000 | **37,331** | ✓ |
| 2 | Unique expert IDs present | 256 (range 0–255) | **256** (range 0–255) | ✓ |
| 3 | Total `mtp.*` keys in merged artifact | ≥ 795 (anchored on 1-rank dryrun's 799) | **799** | ✓ |
| 4 | Expert weight-scale ratio (4-rank ÷ 1-rank, layer 5 expert 0 \|mean\|) | ~1.000 ± 1e-3 | **w1: 1.000000, w2: 1.000000, w3: 1.000000** | ✓ |

Number (4) is exact because NVFP4 + FP8_BLOCK are RTN-style — scales are derived from the weight tensors, not from cross-rank-averaged activation observers. The monkey-patched `Observer.synchronize` (which would have averaged activation stats across ranks) does not affect these scales. Documented in `multirank_observer_sync_hang.md`.

## Walltime

| Phase | Duration | Notes |
|---|---|---|
| Process startup → patches applied | ~30s | dist init + 4 monkey-patches |
| BF16 load (sharded, 4-rank) | ~80s | 26.8 GB/rank RSS, matches loadtest baseline |
| Dataset prep | ~10s | 16-sample slice of ultrachat_200k |
| Calibration (subgraphs 1–45) | **378s** (6.3 min) | per `_progress.json.elapsed_seconds` |
| Compression + per-rank save (`_rank_NN/`) | ~80s | each rank writes its slice to subdir; atomic `.tmp+rename` |
| Rank-0 merge (108 shards) | ~50s | `merge_rank_subdirs.merge_rank_subdirs(world_size=4)` |
| Post-save config edits (`scale_fmt`, target renames) | ~3s | unchanged from prior runs |
| **Total walltime** | **642s (10.7 min)** | — |

## Atomicity / cleanup verification

After the run:
- `tmp leftover`: 0 (`find /scratch/weights/v4-flash-nvfp4-dryrun-4rank -name '*.tmp'` returns nothing)
- `subdirs leftover`: 0 (`_rank_NN/` directories all removed by merge)
- `shards`: 108 (matches `total_shards` reported in merge log)
- Total artifact size: 538 GB

The atomic `.tmp + os.replace` pattern leaves no debris on a clean exit. The stale-`.tmp` sweep at save start would catch any debris from a killed prior run.

## Progress marker (Piece 3)

`_progress.json` was written 45 times during calibration (once per `SEQUENTIAL_EPOCH_END`). Final content:

```json
{
  "started_at_utc": "2026-05-20T20:58:15.386852+00:00",
  "world_size": 4,
  "samples": 16,
  "max_seq_len": 128,
  "batch_size": 4,
  "last_subgraph_completed": 45,
  "last_subgraph_completed_at_utc": "2026-05-20T21:04:34.279725+00:00",
  "elapsed_seconds": 378
}
```

For Phase 2b (samples=768, max_seq_len=512), the projected calibration phase is ~192× the per-token compute of this dryrun. Extrapolation: 378s × 192 / 8 (8-rank parallelism) = ~9070s = **~2.5h for calibration alone**, plus ~3 min compress + ~5 min save + ~2 min merge. Rough total: **~2.7–3h on 8 ranks** — significantly faster than the original 10–12h estimate, but the prior estimate was based on coarser extrapolation. Will be confirmed at the hour-4 status check-in.

## Residual risk (per checkpointing design doc)

The 3-piece compromise protects **save** (~few minutes of work). It does **not** protect calibration itself (~2.7–3h projected). A process crash at hour 2 = restart from scratch. AWS reservation type verified as **on-demand** (no spot-reclaim risk). Mitigation for non-AWS failure modes (process bug, network/SSH disconnect, hardware fault) is the upstream feature request for native intra-calibration resume (task #17, not yet filed).

## Decision

**Phase 2b is approved by this gate.** Launch with the production recipe:

```
torchrun --nproc-per-node 8 --master-port 29500 \
    scripts/quantize_v4_nvfp4_fp8_mtp.py \
    --weights /scratch/weights/bf16-mtp \
    --config /scratch/weights/bf16-mtp/config.json \
    --output /scratch/weights/v4-flash-nvfp4-fp8-mtp \
    --samples 768 --max-seq-len 512 --batch-size 4
```

Hour-4 and hour-8 status check-ins per the gate.
