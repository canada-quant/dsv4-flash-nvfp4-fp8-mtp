#!/usr/bin/env python3
"""Summarize the quantization scale tensors in a dryrun artifact.

Used as A/B verification for the observer-sync monkey-patch: load both the
1-rank and 4-rank-with-patch dryrun outputs, print scale magnitude/shape per
target Linear, and confirm the multi-rank case isn't producing degenerate
(all-zero, infinite, NaN) scales.

Usage::

    python scripts/inspect_dryrun_scales.py /scratch/weights/v4-flash-nvfp4-dryrun
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", help="dryrun output dir")
    args = ap.parse_args()

    artifact = Path(args.artifact)
    if not artifact.exists():
        print(f"FATAL: {artifact} does not exist", file=sys.stderr)
        sys.exit(1)

    shards = sorted(artifact.glob("*.safetensors"))
    if not shards:
        print(f"FATAL: no safetensors in {artifact}", file=sys.stderr)
        sys.exit(1)

    print(f"[inspect] artifact: {artifact}")
    print(f"[inspect] shards: {len(shards)}")

    scale_keys = []
    weight_keys = []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith("_scale") or k.endswith("_global_scale"):
                    scale_keys.append((k, shard))
                elif k.endswith(".weight"):
                    weight_keys.append((k, shard))

    print(f"[inspect] scale tensors: {len(scale_keys)}")
    print(f"[inspect] weight tensors: {len(weight_keys)}")
    print()

    bad = []
    for k, shard in sorted(scale_keys):
        with safe_open(shard, framework="pt", device="cpu") as f:
            t = f.get_tensor(k)
        finite_mask = t.isfinite()
        n = t.numel()
        n_finite = int(finite_mask.sum().item())
        n_zero = int((t == 0).sum().item())
        if n_finite < n:
            bad.append((k, f"non-finite {n - n_finite}/{n}"))
        elif n_zero == n:
            bad.append((k, "all zero"))
        finite_t = t[finite_mask]
        if finite_t.numel() == 0:
            mn = mx = float("nan")
        else:
            mn = float(finite_t.abs().min().item())
            mx = float(finite_t.abs().max().item())
        print(
            f"  {k:80s} shape={tuple(t.shape)} dtype={t.dtype} "
            f"|min|={mn:.3e} |max|={mx:.3e} zero={n_zero}/{n}"
        )

    if bad:
        print()
        print("=" * 60)
        print(f"DEGENERATE SCALES ({len(bad)}):")
        for k, why in bad:
            print(f"  {k}: {why}")
        sys.exit(2)

    print()
    print("[inspect] all scales finite and non-degenerate")


if __name__ == "__main__":
    main()
