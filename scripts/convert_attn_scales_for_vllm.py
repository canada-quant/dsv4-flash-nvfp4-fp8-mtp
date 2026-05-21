#!/usr/bin/env python3
"""Convert attn weight_scale tensors to vLLM-mainline DSV4 expected format.

Discovery 2026-05-21 Phase 5a: vLLM mainline DSV4 attention forward at
`vllm/models/deepseek_v4/attention.py:334` reads `self.wo_a.weight_scale_inv`,
and the DeepGemm kernel's `process_weights_after_loading` at
`vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py:78-89` uses
the parameter named `weight_scale_inv` if present (else falls back to
`weight_scale` but registers the post-processed result back to the same
name it found). The DeepGemm-internal block-scale post-processor
`deepgemm_post_process_fp8_weight_block` (at `vllm/model_executor/layers/
quantization/utils/fp8_utils.py:1017`) asserts that the input scale dtype
is `torch.float32` (or `torch.float8_e8m0fnu`); BF16 is rejected.

Llm-compressor's FP8_BLOCK preset saves attn `.weight_scale` as bfloat16.
Our calibration recipe inherited this. To serve cleanly through the
DeepGemm + DSV4-attention forward path, we need:

  1. Key rename: `*.attn.*.weight_scale` -> `*.attn.*.weight_scale_inv`
     (does NOT apply to FFN expert keys or MTP keys — only attn.)
  2. Dtype upcast: BF16 -> FP32 on those same keys (~33k tensors total
     for V4-Flash, ~80 attn Linears x ~400 sites). Numeric values identical
     (BF16 -> FP32 is lossless upcast).
  3. Config edit: add input_activations to config_groups[group_0] matching
     RedHat's dynamic FP8 group=128 spec. This makes vLLM pick the W8A8Fp8
     scheme which routes through DeepGemm (with is_bmm BMM reshape support)
     instead of W8A16Fp8 (Marlin only, no BMM reshape).

This is purely an artifact-side postprocess — no re-calibration. Atomic
.tmp + rename per safetensor shard.

The MTP block (Option Y, unquantized BF16) has no weight_scale tensors so
is unaffected by step 1/2. Its inclusion at config_groups level is
similarly N/A — MTP is in the `ignore` list.

Safe to re-run; idempotent (checks current name + dtype before changing).
"""
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", help="artifact dir with model.safetensors.index.json")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without modifying anything",
    )
    args = ap.parse_args()

    art = Path(args.artifact)
    idx_path = art / "model.safetensors.index.json"
    cfg_path = art / "config.json"

    # ---- Step 1+2: rename + dtype upcast safetensors ----
    idx = json.loads(idx_path.read_text())
    weight_map = idx["weight_map"]

    # Match the same attn.* prefix patterns we have in config_groups[group_0]
    # targets, AFTER the fused-name update from update_config_for_fused_attn.py.
    # We need to catch the on-disk UNFUSED names since fusing happens at vLLM
    # load time (the artifact stays unfused on disk).
    attn_scale_pat = re.compile(
        r"^(layers\.\d+|mtp\.\d+)\.attn"
        r"(?:\.compressor)?"  # optional compressor sub-block
        r"\.(wq_a|wq_b|wkv|wgate|wo_a|wo_b)\.weight_scale$"
    )

    # Group keys by shard so we rewrite each shard once.
    by_shard: dict[str, list[tuple[str, str]]] = {}  # shard -> [(old_key, new_key)]
    for k in list(weight_map.keys()):
        if attn_scale_pat.match(k):
            new_k = k[:-len(".weight_scale")] + ".weight_scale_inv"
            by_shard.setdefault(weight_map[k], []).append((k, new_k))

    total = sum(len(v) for v in by_shard.values())
    print(f"attn weight_scale keys to rename + upcast: {total}")
    print(f"shards touched: {len(by_shard)}")
    if args.dry_run:
        for shard, pairs in list(by_shard.items())[:3]:
            print(f"  {shard}: {len(pairs)} keys")
            for o, n in pairs[:3]:
                print(f"    {o} -> {n}")
        return

    if total == 0:
        print("Nothing to rename — artifact may already be converted.")
    else:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        idx_backup = idx_path.with_suffix(f".json.bak_pre_attn_rename_{ts}")
        shutil.copy2(idx_path, idx_backup)
        print(f"index backup: {idx_backup}")

        for shard, pairs in sorted(by_shard.items()):
            sp = art / shard
            tmp = sp.with_suffix(sp.suffix + ".tmp")
            tensors = {}
            renames = dict(pairs)
            with safe_open(sp, framework="pt", device="cpu") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    new_k = renames.get(k, k)
                    if k in renames:
                        # BF16 -> FP32 upcast for renamed keys (DeepGemm
                        # rejects BF16 scales at process_weights_after_loading).
                        if t.dtype == torch.bfloat16:
                            t = t.to(torch.float32)
                        elif t.dtype != torch.float32:
                            print(
                                f"  WARN: {k} dtype is {t.dtype}, expected "
                                "bfloat16 — leaving as-is"
                            )
                    tensors[new_k] = t
            save_file(tensors, str(tmp), metadata={"format": "pt"})
            os.replace(tmp, sp)
            print(f"  rewrote {shard}: {len(pairs)} keys renamed+upcast")

        # Update index
        new_weight_map = {}
        for k, shard in weight_map.items():
            pairs = by_shard.get(shard, [])
            renamed = dict(pairs)
            new_k = renamed.get(k, k)
            new_weight_map[new_k] = shard
        idx["weight_map"] = new_weight_map
        idx_tmp = idx_path.with_suffix(".json.tmp")
        idx_tmp.write_text(json.dumps(idx, indent=2))
        os.replace(idx_tmp, idx_path)
        print(f"updated: {idx_path}")

    # ---- Step 3: config — add input_activations to group_0 ----
    cfg = json.loads(cfg_path.read_text())
    g0 = cfg.get("quantization_config", {}).get("config_groups", {}).get("group_0")
    if g0 is None:
        raise SystemExit("FATAL: config_groups.group_0 not present")

    expected = {
        "actorder": None,
        "block_structure": None,
        "dynamic": True,
        "group_size": 128,
        "num_bits": 8,
        "observer": None,
        "observer_kwargs": {},
        "scale_dtype": None,
        "strategy": "group",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    }
    if g0.get("input_activations") == expected:
        print("config_groups.group_0.input_activations already set — skipping config update.")
    else:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        cfg_backup = cfg_path.with_suffix(
            f".json.bak_pre_input_activations_{ts}"
        )
        shutil.copy2(cfg_path, cfg_backup)
        g0["input_activations"] = expected
        tmp = cfg_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, indent=2))
        os.replace(tmp, cfg_path)
        print(f"updated config_groups.group_0.input_activations: dynamic FP8 group=128")
        print(f"  config backup: {cfg_backup}")


if __name__ == "__main__":
    main()
