#!/usr/bin/env python3
"""Unified Phase 3 postprocess pipeline for V4-Flash-NVFP4-FP8-MTP artifacts.

Runs all postprocess steps in the correct order so an artifact produced by
`quantize_v4_nvfp4_fp8_mtp{,_v2}.py` becomes vLLM-mainline-loadable:

  1. `postprocess_for_vllm.py` (main Phase 3) — rename mtp embedding, set
     num_hidden_layers=43, set expert_dtype=fp4, inject scale_fmt: ue8m0,
     inject packed_modules_mapping. **This is the existing Phase 3 driver.**
  2. `update_config_for_fused_attn.py` — rewrite
     `config_groups[group_0].targets` from unfused (`wq_a|wkv|...`) to
     FUSED (`fused_wqa_wkv|compressor.fused_wkv_wgate|wq_b|wo_a|wo_b`)
     names that match vLLM's MergedColumnParallelLinear prefix.
  3. `convert_attn_scales_for_vllm.py` — upcast attn `.weight_scale` BF16
     → FP32 (DeepGemm requirement) AND inject `input_activations` (dynamic
     FP8 group=128) into `config_groups[group_0]` matching RedHat's spec.
  4. `squeeze_global_scales.py` — squeeze `(1,)`-shape global_scale tensors
     to 0-D scalar (vLLM's MoE loader requires scalar).
  5. (v2 only) Add MTP-layer regex to the config's `ignore` list so vLLM's
     MTP draft model construction doesn't apply NVFP4 quantization to the
     MTP block's FFN experts (which would allocate `w13_weight_packed`
     while the loader expects `w13_weight`). Pattern:
     `re:.*\\.layers\\.{num_hidden_layers}\\..*` — matches the MTP layer
     index that vLLM uses internally.

Each step has its own backup + atomic write. Idempotent — re-running is
safe.

Usage:
  python postprocess_v2_pipeline.py /scratch/weights/v4-flash-nvfp4-fp8-mtp-v2
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'='*70}\n[STEP {name}] {' '.join(cmd)}\n{'='*70}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        sys.exit(f"FATAL: step {name} exited {res.returncode}")


def add_mtp_layer_ignore(artifact: Path) -> None:
    """Step 5: add MTP-layer regex to ignore list."""
    cfg_path = artifact / "config.json"
    cfg = json.loads(cfg_path.read_text())
    qc = cfg.get("quantization_config", {})
    ig = qc.get("ignore", [])
    num_hidden = cfg.get("num_hidden_layers", 43)

    new_entries = [
        rf"re:.*\.layers\.{num_hidden}\..*",
        rf"re:.*\.layers\.{num_hidden}$",
    ]
    added = []
    for entry in new_entries:
        if entry not in ig:
            ig.append(entry)
            added.append(entry)

    if not added:
        print(f"[STEP 5] MTP-layer ignore already present — skipping")
        return

    backup = cfg_path.with_suffix(
        f".json.bak_pre_mtp_ignore_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    shutil.copy2(cfg_path, backup)

    qc["ignore"] = ig
    tmp = cfg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, cfg_path)
    print(
        f"[STEP 5] added {len(added)} MTP-layer ignore entries "
        f"(num_hidden={num_hidden}); backup: {backup.name}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", help="artifact dir from quantize_v4_*.py")
    ap.add_argument(
        "--skip-mtp-ignore",
        action="store_true",
        help="skip step 5 (use for v1 artifacts where MTP is fully BF16 and "
        "the draft model isn't expected to load anyway)",
    )
    args = ap.parse_args()

    artifact = Path(args.artifact)
    if not artifact.exists():
        sys.exit(f"FATAL: artifact dir {artifact} does not exist")

    # Step 1: main postprocess (rename mtp embed, scale_fmt, expert_dtype etc.)
    run_step(
        "1/postprocess_for_vllm",
        [
            sys.executable,
            str(SCRIPT_DIR / "postprocess_for_vllm.py"),
            str(artifact),
        ],
    )

    # Step 2: fused-attn regex
    run_step(
        "2/update_config_for_fused_attn",
        [
            sys.executable,
            str(SCRIPT_DIR / "update_config_for_fused_attn.py"),
            str(artifact),
        ],
    )

    # Step 3: scale dtype + input_activations + W8A8 config
    run_step(
        "3/convert_attn_scales_for_vllm",
        [
            sys.executable,
            str(SCRIPT_DIR / "convert_attn_scales_for_vllm.py"),
            str(artifact),
        ],
    )

    # Step 4: squeeze global_scale (1,) → 0-d
    run_step(
        "4/squeeze_global_scales",
        [
            sys.executable,
            str(SCRIPT_DIR / "squeeze_global_scales.py"),
            str(artifact),
        ],
    )

    # Step 5: MTP-layer ignore for draft-model construction
    if args.skip_mtp_ignore:
        print("\n[STEP 5] skipped (--skip-mtp-ignore)")
    else:
        print(f"\n{'='*70}\n[STEP 5] add_mtp_layer_ignore\n{'='*70}", flush=True)
        add_mtp_layer_ignore(artifact)

    print("\n=" * 35)
    print(f"\nPostprocess pipeline complete for: {artifact}")
    print("\nNext: Phase 4 — run scripts/verify_mtp_quantized.py "
          "to confirm MTP retention invariants.")


if __name__ == "__main__":
    main()
