#!/usr/bin/env python3
"""Update config.json's group_0 regex to match vLLM mainline's FUSED attn names.

Discovery 2026-05-21 Phase 5a: vLLM mainline DSV4 (vllm/models/deepseek_v4/
nvidia/model.py:871) uses MergedColumnParallelLinear for `attn.fused_wqa_wkv`
(merging wq_a + wkv) and `attn.compressor.fused_wkv_wgate` (merging wkv +
wgate). Its `load_weights` has a `stacked_params_mapping` that renames
on-disk unfused keys to the fused merged-Linear name at load time.

For the rename to find a target slot for `.weight_scale`, the merged Linear
must be QUANTIZED at construction. The quant framework only allocates a
`weight_scale` parameter on a module whose prefix matches the
`config_groups[*].targets` regex. Our calibration recipe targeted UNFUSED
prefixes (`re:.*\\.attn\\.(wq_a|wq_b|wkv|wo_a|wo_b)$`), so the merged Linear
was NOT considered quantized at vLLM construction, and the rename target
slot didn't exist:

  KeyError: 'layers.0.attn.fused_wqa_wkv.weight_scale'

The fix is purely config-side: rewrite the regex to use the FUSED merged-
Linear prefixes. The on-disk tensors stay unfused; vLLM's load-time stacked-
params mapping does the concat into the merged param.

This is reversible — config.json is backed up to a timestamped sibling file.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

OLD_TARGETS = [r"re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$"]
NEW_TARGETS = [
    r"re:.*\.attn\.(fused_wqa_wkv|compressor\.fused_wkv_wgate|wq_b|wo_a|wo_b)$"
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", help="artifact dir containing config.json")
    args = ap.parse_args()

    art = Path(args.artifact)
    cfg_path = art / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"FATAL: {cfg_path} not found")

    backup = cfg_path.with_suffix(
        f".json.bak_pre_fused_attn_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    shutil.copy2(cfg_path, backup)
    print(f"backup: {backup}")

    c = json.loads(cfg_path.read_text())
    qc = c.get("quantization_config", {})
    cg = qc.get("config_groups", {})
    g0 = cg.get("group_0")
    if g0 is None:
        raise SystemExit("FATAL: config_groups.group_0 not present")

    old = g0.get("targets")
    print(f"old group_0 targets: {old}")
    if old == NEW_TARGETS:
        print("Already using fused targets — nothing to do.")
        return

    g0["targets"] = NEW_TARGETS
    print(f"new group_0 targets: {NEW_TARGETS}")

    tmp = cfg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(c, indent=2))
    os.replace(tmp, cfg_path)
    print(f"wrote: {cfg_path}")


if __name__ == "__main__":
    main()
