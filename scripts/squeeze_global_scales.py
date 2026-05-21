"""Post-process: squeeze (1,)-shape global_scale tensors to scalar (0-d).

vLLM's MoE loader at `fused_moe/layer.py:_load_per_tensor_weight_scale` does
`target = param_data[expert_id][idx]` which gives a scalar (0-D) target,
then `target.copy_(loaded_weight)` — torch's broadcast check is strict on
the in-place op and rejects loading a shape `(1,)` tensor into a `()`
target with "output with shape [] doesn't match the broadcast shape [1]".

Our llmcompressor save emits `weight_global_scale` and `input_global_scale`
as shape `(1,)`. Sweep all such tensors in the artifact and squeeze the
1-d singletons to 0-d. Atomic .tmp + rename per shard.

This is a vLLM-side bug class too (the loader should `.view([]).copy_(...)`
or wrap with `.squeeze()`); filed as a follow-up upstream task. For now,
artifact-side fix is the unblock.
"""
import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", help="artifact dir with model.safetensors.index.json")
    args = ap.parse_args()

    ART = Path(args.artifact)
    with open(ART / "model.safetensors.index.json") as f:
        idx = json.load(f)
    wmap = idx["weight_map"]

    global_scale_keys = [k for k in wmap if "_global_scale" in k]
    print(f"global_scale keys to inspect: {len(global_scale_keys)}")

    needs_squeeze_by_shard: dict[str, list[str]] = {}
    for k in global_scale_keys:
        shard = wmap[k]
        with safe_open(ART / shard, framework="pt", device="cpu") as f:
            t = f.get_tensor(k)
        if tuple(t.shape) == (1,):
            needs_squeeze_by_shard.setdefault(shard, []).append(k)

    print(f"shards with 1-d global scales: {len(needs_squeeze_by_shard)}")
    if not needs_squeeze_by_shard:
        print("Nothing to squeeze — exiting")
        return

    total_squeezed = 0
    for shard, keys in sorted(needs_squeeze_by_shard.items()):
        sp = ART / shard
        tmp = sp.with_suffix(sp.suffix + ".tmp")
        tensors = {}
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k in keys and tuple(t.shape) == (1,):
                    t = t.squeeze()
                    total_squeezed += 1
                tensors[k] = t
        save_file(tensors, str(tmp), metadata={"format": "pt"})
        os.replace(tmp, sp)
        print(f"  {shard}: squeezed {len(keys)} keys")

    print(f"total squeezed: {total_squeezed}")

    # Verify
    any_shard = next(iter(needs_squeeze_by_shard))
    with safe_open(ART / any_shard, framework="pt", device="cpu") as f:
        for k in needs_squeeze_by_shard[any_shard][:3]:
            t = f.get_tensor(k)
            print(f"  verify: {k}: shape={tuple(t.shape)} dtype={t.dtype}")


if __name__ == "__main__":
    main()
