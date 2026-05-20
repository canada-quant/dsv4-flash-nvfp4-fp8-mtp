#!/usr/bin/env python3
"""Phase 2 — NVFP4-FP8 calibration with MTP layer preserved.

The DeepSeek-V4-Flash NVFP4 recipe that **keeps** the MTP speculative-decoding
head. RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` ships the same math but
drops MTP because stock HF transformers' `DeepseekV4PreTrainedModel` has
`_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`. Our edge is the
vendored upstream model + `patches/modeling_deepseek_v4.py.diff`.

Why this is the cheap path:
  - `QuantizationModifier` (used here) is RTN-style — no Hessian, no GPTQ
    iterative column adjustment. Forward through calibration data just sets
    observer min/max statistics; quantization is then weight-only.
  - The B300 NCCL+NVLink hang we hit on `GPTQModifier._reduce_hessian_to_target_rank`
    is GPTQ-specific. `QuantizationModifier` in llmcompressor has zero `dist.*`
    calls in its main file (verified 2026-05-20). **Still verify with a
    multi-rank dryrun** — observer code or compressed_tensors internals could
    have collectives we haven't audited.

Recipe (matches RedHat NVFP4 + adds MTP):
  - Routed MoE experts: NVFP4 (4-bit float, group_size=16, tensor_group,
    FP8-e4m3 scales). Format: `nvfp4-pack-quantized`.
  - Attention Linears (wq_a/wq_b/wkv/wo_a/wo_b): FP8_BLOCK 128x128.
  - MTP entry projections (mtp.0.e_proj, mtp.0.h_proj): FP8_BLOCK 128x128.
  - Everything else (norms, gates, shared experts, hc_*, attn.compressor,
    attn.indexer, attn_sink, embed, head): ignored, stays BF16.

Single-process::

    python scripts/quantize_v4_nvfp4_fp8_mtp.py \\
        --weights /scratch/weights/bf16-mtp \\
        --config /scratch/weights/bf16-mtp/config.json \\
        --output /scratch/weights/v4-flash-nvfp4-fp8-mtp \\
        --samples 768 --max-seq-len 512 --batch-size 4

Multi-process (target topology if dryrun is clean)::

    torchrun --nproc-per-node 8 --master-port 29500 \\
        scripts/quantize_v4_nvfp4_fp8_mtp.py \\
        --weights /scratch/weights/bf16-mtp \\
        --config /scratch/weights/bf16-mtp/config.json \\
        --output /scratch/weights/v4-flash-nvfp4-fp8-mtp \\
        --samples 768 --max-seq-len 512 --batch-size 4

Dryrun gate (REQUIRED before full run)::

    torchrun --nproc-per-node 4 --master-port 29500 \\
        scripts/quantize_v4_nvfp4_fp8_mtp.py ... \\
        --samples 16 --max-seq-len 128 --batch-size 4 --dry-run-one-layer

The dryrun is the gate that proves `QuantizationModifier` doesn't deadlock on
B300 like `GPTQModifier` did. If it completes, the full 8-rank run is safe.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Make scripts.upstream importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import torch.nn as nn

from scripts.upstream import (
    Transformer,
    apply_dist_state,
    build_model_args,
)
from scripts.calibration_model import CalibrationModel
from scripts.load_bf16_into_transformer import load_safetensors_into


# =========================================================================
# V4 manual chat encoding (predecessor recipe, verbatim)
# =========================================================================
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"


def preprocess_v4(example: dict) -> dict:
    text = BOS
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        if role == "system":
            text += content
        elif role == "user":
            text += f"<｜User｜>{content}"
        elif role == "assistant":
            text += f"<｜Assistant｜></think>{content}{EOS}"
    return {"text": text}


# =========================================================================
# PreTrainedModel-bridge config (vLLM-loadable on save)
# =========================================================================
class _CompatConfig:
    """Config that llmcompressor.oneshot reads at runtime AND produces a
    vLLM-loadable ``config.json`` at save time. Reads bf16-mtp/config.json
    as the base and merges our overrides + tokenizer files.
    """

    base_model_prefix = "model"
    is_encoder_decoder = False

    def __init__(self, args, upstream_config_path: str | None = None):
        self.model_type = "deepseek_v4"
        self.tie_word_embeddings = True
        self.hidden_size = args.dim
        self.num_hidden_layers = args.n_layers + args.n_mtp_layers
        self.vocab_size = args.vocab_size
        self.architectures = ["DeepseekV4ForCausalLM"]
        self.torch_dtype = "bfloat16"
        self._upstream_config_path = upstream_config_path

    @property
    def use_return_dict(self):
        return True

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def save_pretrained(self, save_directory, **_kw):
        base: dict = {}
        upstream_dir = None
        if self._upstream_config_path and os.path.exists(self._upstream_config_path):
            with open(self._upstream_config_path) as f:
                base = json.load(f)
            upstream_dir = os.path.dirname(self._upstream_config_path)
        base.pop("quantization_config", None)
        merged = {**base, **self.to_dict()}
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(merged, f, indent=2, default=str)
        if upstream_dir:
            for fname in ("tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "generation_config.json",
                          "chat_template.jinja"):
                src = os.path.join(upstream_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(save_directory, fname))


class _CompatModel(nn.Module):
    """nn.Module wrapper with .config + name_or_path so llmcompressor's
    save_pretrained_wrapper can write config.json + recipe.yaml correctly.
    """

    def __init__(self, calibration_model: CalibrationModel, args,
                 upstream_config_path: str | None = None):
        super().__init__()
        self.model = calibration_model
        self.config = _CompatConfig(args, upstream_config_path)
        # update_and_save_recipe(model.name_or_path, ...) — empty string ok.
        self.name_or_path = ""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_input_embeddings(self):
        return self.model.transformer.embed

    def get_output_embeddings(self):
        return self.model.transformer.head

    def tie_weights(self):
        pass

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input_ids, **kwargs)


# =========================================================================
# Recipe
# =========================================================================
def build_nvfp4_recipe(dry_run_one_layer: bool):
    """Return a QuantizationModifier with NVFP4 experts + FP8_BLOCK attn."""
    from compressed_tensors.quantization import QuantizationScheme
    from compressed_tensors.quantization.quant_scheme import NVFP4, FP8_BLOCK
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if dry_run_one_layer:
        # Restrict to layer 5 (a representative MoE layer) so the dryrun
        # exercises both FP8_BLOCK attn and NVFP4 experts paths quickly.
        attn_targets = [r"re:.*\.layers\.5\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$"]
        expert_targets = [r"re:.*\.layers\.5\.ffn\.experts\.\d+\.(w1|w2|w3)$"]
    else:
        attn_targets = [
            r"re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
            r"re:.*mtp\.\d+\.(e_proj|h_proj)$",
        ]
        expert_targets = [r"re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$"]

    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=attn_targets,
                format="float-quantized",
                **FP8_BLOCK,
            ),
            "experts": QuantizationScheme(
                targets=expert_targets,
                format="nvfp4-pack-quantized",
                **NVFP4,
            ),
        },
        ignore=[
            "head", "embed",
            r"re:.*norm.*",
            r"re:.*\.ffn\.gate$",
            r"re:.*\.ffn\.gate\..*",
            r"re:.*\.ffn\.shared_experts\..*",
            r"re:.*\.hc_.*",
            r"re:hc_.*",
            r"re:.*\.attn\.attn_sink$",
            r"re:.*\.attn\.(compressor|indexer)\..*",
        ],
    )


def build_calibration_dataset(tokenizer, num_samples: int, max_seq_len: int, seed: int = 42):
    """Locked corpus: HuggingFaceH4/ultrachat_200k train_sft seed=42 (predecessor)."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{num_samples * 2}]",
    )
    ds = ds.shuffle(seed=seed)
    ds = ds.map(preprocess_v4)
    ds = ds.select(range(num_samples))

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    rev = None
    try:
        info = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:1]")
        rev = getattr(info, "info", None)
        rev = getattr(rev, "version", None) if rev is not None else None
    except Exception:
        pass
    return ds, rev


# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Phase-1 BF16 dir")
    ap.add_argument("--config", required=True,
                    help="upstream config.json")
    ap.add_argument("--output", required=True, help="output NVFP4-FP8-MTP dir")
    ap.add_argument("--samples", type=int, default=768)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dry-run-one-layer", action="store_true",
                    help="recipe restricted to layer 5 only; gate test for "
                         "multi-rank NCCL on the QuantizationModifier path")
    args = ap.parse_args()

    t_total = time.time()

    # ---- 1. distributed init ---------------------------------------------
    use_dist = "TORCHELASTIC_RUN_ID" in os.environ
    if use_dist:
        # 2h NCCL timeout (vs 10min default). The GPTQ path hit a 10min timeout
        # on a Hessian REDUCE; even though this path doesn't reduce Hessians,
        # the safer default keeps any unaudited collective from killing the run
        # on a soft hang.
        if not dist.is_initialized():
            _rank = int(os.environ["RANK"])
            _local_rank = int(os.environ["LOCAL_RANK"])
            _world_size = int(os.environ["WORLD_SIZE"])
            _device = torch.device(f"cuda:{_local_rank}")
            torch.cuda.set_device(_device)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=_rank,
                world_size=_world_size,
                device_id=_device,
                timeout=_dt.timedelta(hours=2),
            )
            dist.barrier()
    elif not dist.is_initialized():
        # llm-compressor's compress code calls dist.get_rank() unconditionally;
        # init a 1-rank gloo group so single-process invocations work too.
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{free_port}",
            rank=0,
            world_size=1,
        )
    apply_dist_state()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0
    if is_main:
        print(f"[dist] world_size={world_size} rank={rank} use_dist={use_dist}",
              flush=True)

    # ---- 2. ModelArgs from upstream config -------------------------------
    margs = build_model_args(
        args.config, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len
    )
    if is_main:
        print(f"[args] dim={margs.dim} n_layers={margs.n_layers} "
              f"n_mtp_layers={margs.n_mtp_layers} "
              f"n_routed_experts={margs.n_routed_experts}", flush=True)

    # ---- 3. Instantiate Transformer + load BF16 --------------------------
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")

    if is_main:
        print("[load] instantiating Transformer on CPU (init-skip)", flush=True)
    # Mask dist during construction so Transformer.__init__ doesn't re-set
    # upstream world_size from dist (we want it pinned at 1 for non-MoE
    # classes; MoE shards via _expert_world_size separately).
    _orig_is_init = dist.is_initialized
    _orig_get_ws = dist.get_world_size
    _orig_get_rk = dist.get_rank
    dist.is_initialized = lambda: False
    dist.get_world_size = lambda *a, **kw: 1
    dist.get_rank = lambda *a, **kw: 0
    t0 = time.time()
    try:
        transformer = Transformer(margs)
    finally:
        dist.is_initialized = _orig_is_init
        dist.get_world_size = _orig_get_ws
        dist.get_rank = _orig_get_rk
    if is_main:
        print(f"[load] instantiated in {time.time()-t0:.1f}s", flush=True)
        print(f"[load] streaming safetensors from {args.weights}", flush=True)

    t1 = time.time()
    loaded, unmatched, missing = load_safetensors_into(
        transformer, Path(args.weights), verbose=is_main
    )
    if is_main:
        print(f"[load] loaded={loaded} unmatched={len(unmatched)} "
              f"missing={len(missing)} in {time.time()-t1:.1f}s", flush=True)
    # Expert sharding produces expected unmatches (this rank doesn't own them).
    non_expert_unmatched = [
        k for k in unmatched if re.search(r"\.experts\.\d+\.", k) is None
    ]
    if non_expert_unmatched:
        if is_main:
            print(
                f"FATAL: {len(non_expert_unmatched)} unmatched non-expert "
                f"keys: {non_expert_unmatched[:10]}",
                flush=True,
            )
        sys.exit(2)

    # ---- 4. tokenizer + dataset ------------------------------------------
    if is_main:
        print(f"[tokenizer] loading from {args.weights}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.weights, trust_remote_code=False)

    if is_main:
        print(f"[dataset] preparing {args.samples} samples from ultrachat_200k",
              flush=True)
    ds, ds_rev = build_calibration_dataset(
        tokenizer, num_samples=args.samples, max_seq_len=args.max_seq_len
    )
    if is_main:
        print(f"[dataset] {len(ds)} samples ready; revision={ds_rev}", flush=True)

    # ---- 5. wrap model ---------------------------------------------------
    cal = CalibrationModel(transformer)
    model = _CompatModel(cal, margs, upstream_config_path=args.config)

    # ---- 6. recipe + oneshot ---------------------------------------------
    if is_main:
        print(f"[recipe] building NVFP4 recipe (dry_run_one_layer={args.dry_run_one_layer})",
              flush=True)
    recipe = build_nvfp4_recipe(args.dry_run_one_layer)

    if is_main:
        print(f"[oneshot] starting calibration samples={args.samples} "
              f"batch={args.batch_size} seq={args.max_seq_len}", flush=True)
    t_oneshot = time.time()

    try:
        from llmcompressor import oneshot
        oneshot(
            model=model,
            tokenizer=tokenizer,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_seq_len,
            num_calibration_samples=args.samples,
            sequential_targets=["Block"],
            batch_size=args.batch_size,
            output_dir=args.output,
        )
    except Exception as exc:
        is_offload_twice = (
            isinstance(exc, ValueError)
            and "offload a module twice" in str(exc).lower()
        )
        if is_main:
            print(f"[oneshot] failed with: {type(exc).__name__}: {exc}",
                  flush=True)
            if is_offload_twice:
                print("[oneshot] non-fatal: artifact already on disk; "
                      "continuing to post-save processing.", flush=True)
        if not is_offload_twice:
            raise

    # ---- 7. post-save: inject scale_fmt and fix targets ------------------
    # NVFP4 experts use FP8-e4m3 scales; vLLM's DeepseekV4 reads
    # config.quantization_config["scale_fmt"] for the FP8 attn path. Inject.
    if is_main:
        out_cfg = os.path.join(args.output, "config.json")
        if os.path.exists(out_cfg):
            with open(out_cfg) as f:
                _cfg = json.load(f)
            _qc = _cfg.setdefault("quantization_config", {})
            if _qc.get("scale_fmt") is None:
                _qc["scale_fmt"] = "ue8m0"
                print("[post-save] set quantization_config.scale_fmt = ue8m0",
                      flush=True)
            # Recipe targets use upstream names (w1/w2/w3); vLLM's MoE scheme
            # probe checks HF names (gate_proj/up_proj/down_proj). Rewrite.
            for g in _qc.get("config_groups", {}).values():
                tgts = g.get("targets") or []
                g["targets"] = [
                    t.replace("(w1|w2|w3)", "(gate_proj|up_proj|down_proj)")
                    for t in tgts
                ]
            with open(out_cfg, "w") as f:
                json.dump(_cfg, f, indent=2)
            print("[post-save] config.json updated (scale_fmt, HF target names)",
                  flush=True)

    if is_main:
        print(f"[oneshot] done in {time.time()-t_oneshot:.1f}s", flush=True)
        print(f"CALIBRATION_DONE total={time.time()-t_total:.1f}s output={args.output}",
              flush=True)


if __name__ == "__main__":
    main()
