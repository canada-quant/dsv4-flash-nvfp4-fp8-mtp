#!/usr/bin/env python3
"""Merge per-rank `_rank_NN/` subdirs from a multi-rank save into a unified artifact.

The multi-rank calibration in ``quantize_v4_nvfp4_fp8_mtp.py`` writes each rank's
state_dict slice (with the experts that rank owns) to ``<output>/_rank_NN/``.
This script collects those slices, gives them globally unique shard filenames
in the parent dir, and writes a unified ``model.safetensors.index.json``.

It's invoked at the end of multi-rank save by rank 0, but can also be re-run
standalone as a recovery tool if the calibration process is killed AFTER the
per-rank writes complete but BEFORE the merge finishes (e.g. spot reclamation
mid-save). All state needed is on disk in the per-rank subdirs.

Usage (standalone)::

    python scripts/merge_rank_subdirs.py /scratch/weights/v4-flash-nvfp4-fp8-mtp

Usage (programmatic)::

    from scripts.merge_rank_subdirs import merge_rank_subdirs
    merge_rank_subdirs("/scratch/weights/v4-flash-nvfp4-fp8-mtp", world_size=8)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


_RANK_RE = re.compile(r"^_rank_(\d+)$")


def merge_rank_subdirs(save_directory: str | os.PathLike, world_size: int | None = None,
                       verbose: bool = False) -> dict:
    """Combine ``<save_directory>/_rank_*/`` subdirs into a unified artifact.

    Strategy: each per-rank subdir already contains a valid (partial)
    ``model.safetensors.index.json`` plus its shards. Move each subdir's
    shards into the parent dir under a globally unique shard name, then
    union their weight maps into one unified index.

    Idempotent: if no ``_rank_*/`` subdirs exist, returns immediately.

    :param save_directory: parent dir containing _rank_NN/ subdirs
    :param world_size: expected rank count; if set, asserts that many subdirs
    :param verbose: print progress
    :return: the merged index dict
    """
    save_dir = Path(save_directory)
    rank_dirs = sorted(
        p for p in save_dir.iterdir()
        if p.is_dir() and _RANK_RE.match(p.name)
    )
    if not rank_dirs:
        if verbose:
            print(f"[merge] no _rank_*/ subdirs in {save_dir}; nothing to merge")
        merged_idx_path = save_dir / "model.safetensors.index.json"
        if merged_idx_path.exists():
            with open(merged_idx_path) as f:
                return json.load(f)
        return {}

    if world_size is not None and len(rank_dirs) != world_size:
        raise RuntimeError(
            f"[merge] expected {world_size} rank dirs, found {len(rank_dirs)}: "
            f"{[p.name for p in rank_dirs]}"
        )

    if verbose:
        print(f"[merge] found {len(rank_dirs)} rank dirs in {save_dir}")

    # First pass: read each rank's index, accumulate totals
    all_slices = []
    merged_total_size = 0
    total_shards = 0
    for rd in rank_dirs:
        idx_path = rd / "model.safetensors.index.json"
        if not idx_path.exists():
            raise RuntimeError(f"[merge] {idx_path} missing")
        with open(idx_path) as f:
            r_idx = json.load(f)
        rank_shards = sorted(set(r_idx["weight_map"].values()))
        merged_total_size += r_idx["metadata"]["total_size"]
        total_shards += len(rank_shards)
        all_slices.append((rd, r_idx, rank_shards))
        if verbose:
            print(f"  {rd.name}: {len(r_idx['weight_map'])} keys "
                  f"across {len(rank_shards)} shards")

    # Second pass: rename + move shards into save_dir with globally unique names
    merged_weight_map: dict[str, str] = {}
    shard_idx = 1
    for rd, r_idx, rank_shards in all_slices:
        shard_remap: dict[str, str] = {}
        for old_shard in rank_shards:
            new_shard = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
            src = rd / old_shard
            dst = save_dir / new_shard
            if dst.exists():
                raise RuntimeError(
                    f"[merge] destination shard {dst} already exists; refusing "
                    "to overwrite. clean save_directory before retrying."
                )
            shutil.move(str(src), str(dst))
            shard_remap[old_shard] = new_shard
            shard_idx += 1
        for key, old_s in r_idx["weight_map"].items():
            if key in merged_weight_map:
                raise RuntimeError(
                    f"[merge] duplicate key across ranks: {key!r} "
                    f"present in both {merged_weight_map[key]!r} and the new "
                    f"slice from {rd.name}. expert sharding should produce "
                    "DISJOINT key sets — investigate."
                )
            merged_weight_map[key] = shard_remap[old_s]
        # Clean up the now-empty rank subdir (only index.json remains)
        rd_idx_path = rd / "model.safetensors.index.json"
        if rd_idx_path.exists():
            rd_idx_path.unlink()
        try:
            rd.rmdir()
        except OSError as e:
            if verbose:
                print(f"  warning: could not rmdir {rd}: {e}")

    merged_idx = {
        "metadata": {"total_size": merged_total_size},
        "weight_map": merged_weight_map,
    }
    out_path = save_dir / "model.safetensors.index.json"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(merged_idx, f, indent=2)
    os.replace(tmp_path, out_path)
    if verbose:
        print(f"[merge] wrote unified index with {len(merged_weight_map)} keys "
              f"across {total_shards} shards to {out_path}")
    return merged_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("save_directory")
    ap.add_argument("--world-size", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    merge_rank_subdirs(
        args.save_directory, world_size=args.world_size, verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
