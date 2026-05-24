# Serving the NVFP4+FP8+MTP artifact on RTX PRO 6000 Blackwell (SM 12.0)

The headline numbers in the model card and README are measured on 4× B300
SXM6 (SM 10.3). This document records what was needed to also serve the
same artifact on **RTX PRO 6000 Blackwell Server Edition** (SM 12.0, 96 GiB
HBM per GPU), which is the consumer/server Blackwell family and a much more
accessible target than B300.

Validated 2026-05-23 on a Brev `familiar-teal-worm` instance (4× RTX PRO
6000 Blackwell, driver 580.159, CUDA 12.9 + cuda-toolkit-13-0). The build
chain mirrors what the sibling W4A16+FP8+MTP repo uses on the same
hardware; only the NVFP4-specific differences vs the W4A16 path are
called out below.

## Result summary

Both TP=2 (2 GPUs) and TP=4 (4 GPUs) serve, with CUDA graphs enabled (no
`--enforce-eager`). Numbers below are per-replica.

| Config | bs=1 output tok/s | bs=4 output tok/s | bs=16 output tok/s | bs=1 TPOT median | MTP acceptance | GSM8K-50 strict |
|---|---|---|---|---|---|---|
| TP=2, MTP on | 94.6 | 218.5 | 360.5 | 9.05 ms | 70.3–72.8% | 88% |
| TP=4, MTP on | 101.0 | 254.0 | 440.1 | 8.20 ms | 67.1–74.7% | 90% |

Same harness as the W4A16 sibling repo (`vllm bench serve`, random
256-in/256-out, MTP `num_speculative_tokens=1`). At bs=16, TP=4 is
**1.22× faster per-replica** than TP=2 on this artifact — the opposite
of what was measured on B300 (where TP=4 was faster than TP=8 due to
tensor-core underutilization). RTX PRO 6000's slower PCIe interconnect
plus lower per-GPU compute means the extra parallelism still pays off
at all batch sizes we measured.

For comparison, the W4A16+FP8+MTP sibling on the same RTX PRO 6000 box
measured 98.83 tok/s at TP=2 bs=1 (TPOT 8.55 ms, 71.4% MTP acceptance) —
the two artifacts deliver equivalent decode throughput on this hardware,
with NVFP4 trading ~4% per-replica throughput for ~10% smaller on-disk
footprint (172 GB vs 159 GB).

## Hardware/driver

- 4× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB HBM each
- Compute capability `(12, 0)` — RTX PRO 6000 reports `sm_120`, NOT `sm_103a`
  (B300) nor `sm_100a` (B200). Build vLLM with `TORCH_CUDA_ARCH_LIST=12.0a`.
- Driver 580.159, CUDA 12.9 toolkit pre-installed at `/usr/local/cuda`
- No NVLink between RTX PRO 6000 GPUs. PCIe-only NCCL. Custom AR
  crashes with CUDA invalid-argument, so `--disable-custom-all-reduce`
  is mandatory.
- 96 vCPU, 1 TiB host RAM, 7.6 TiB ephemeral NVMe at `/opt/dlami/nvme`.

## vLLM build

Use the same SM 12.0-tuned vLLM branch the W4A16 sibling uses:

```bash
# Branch: jasl/vllm@ds4-sm120-preview-dev (post-refactor, includes MTP refactor)
git clone -b ds4-sm120-preview-dev https://github.com/jasl/vllm ~/src/vllm
cd ~/src/vllm
TORCH_CUDA_ARCH_LIST="12.0a" \
  pip install --no-build-isolation -v --no-deps .
```

Then apply the same 4 standard cherry-picks documented in
`docs/VLLM_SETUP_ISSUES.md` (PRs #43248, #43288, #43290, #43319), plus
the `packed_modules_mapping` patches in
`scripts/patch_v4_forcausal_packed_mapping.py` and
`scripts/patch_mtp_packed_mapping.py`.

## Three additional patches required for NVFP4 on SM 12.0

These are NEW for NVFP4-on-SM12; the W4A16 sibling doesn't hit them
because W4A16 attention and W4A16 MoE go through different kernels.

### 1. NVFP4 MoE backend selector + `VLLM_TEST_FORCE_FP8_MARLIN=1`

vLLM's NVFP4 MoE backend oracle (`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`)
filters available backends to `FLASHINFER_TRTLLM` only when the model
declares `swiglu_limit` (DSV4-Flash sets `swiglu_limit=10.0`). On
SM 12.0 there's no `FLASHINFER_TRTLLM` NVFP4 MoE kernel that auto-selects
(the SM 12.x kernel is `FLASHINFER_B12X`, but it's intentionally excluded
from auto-selection per the upstream code comment), so the selector
raises `NotImplementedError: No NvFp4 MoE backend supports the deployment
configuration.`

Workaround: set `VLLM_TEST_FORCE_FP8_MARLIN=1`. This bypasses the
`swiglu_limit` filter and forces the Marlin MoE backend, which DOES
support SM 12.0. Same flag works on B300 for an unrelated reason.

### 2. `weight_scale_inv`-or-`weight_scale` fallback in Marlin scaled_mm

After patch 1 forces Marlin for FP8_BLOCK attention layers, Marlin's
`scaled_mm/marlin.py:process_weights_after_loading` reads
`layer.weight_scale_inv`. Our artifact (and RedHat's reference) save FP8
scales as `weight_scale` (no `_inv` suffix), so this crashes with
`AttributeError: 'MergedColumnParallelLinear' object has no attribute
'weight_scale_inv'. Did you mean: 'weight_scale'?`

The existing `wo_a` fallback in `vllm/models/deepseek_v4/attention.py`
(PR #43290) doesn't cover this site. Apply the same fallback pattern in
two places in `marlin.py`:

```python
# In process_weights_after_loading (block_quant branch)
ws = getattr(layer, "weight_scale_inv", None)
if ws is None:
    ws = layer.weight_scale
# ...
target = "weight_scale_inv" if hasattr(layer, "weight_scale_inv") else "weight_scale"
replace_parameter(layer, target, weight_scale_inv.data)

# In apply_weights (block_quant branch)
weight_scale = getattr(layer, "weight_scale_inv", None)
if weight_scale is None:
    weight_scale = layer.weight_scale
```

See `patches/sm120_marlin_scale_inv_fallback.diff` in this repo for the
exact diff.

### 3. Skip Marlin pre-processing for DSV4 `is_bmm=True` layers

This is the patch that distinguishes the NVFP4+SM12 path from everything
else. DSV4-Flash's `wo_a` (and `wo_b`, and `compressor.wkv`) are
ColumnParallel linears tagged `is_bmm=True` because the SM 12.0 attention
forward path calls them via a custom Triton `fp8_einsum` kernel at
`vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py`, NOT via a regular
Marlin matmul.

With patches 1+2 in place, Marlin's `process_weights_after_loading`
repacks the FP8 weights into Marlin's tile layout. The `fp8_einsum`
kernel then reads the layer's `weight` attribute and finds the
tile-packed bytes instead of the original `(N, K)` FP8 layout it
expects. Symptom:

```
RuntimeError: DeepSeek V4 fp8 einsum weight rows must be divisible by
out_rank=1024, got 256
```

(256 = `N / 16` at TP=2 — the Marlin tile row count for a 4096-row
weight.)

Fix: intercept the FP8_BLOCK scheme's `process_weights_after_loading`
and skip the Marlin repack for layers tagged `is_bmm=True`. The
`fp8_einsum` kernel handles the original layout directly, no
pre-processing required. Patch in
`vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`:

```python
elif self.strategy == QuantizationStrategy.BLOCK:
    assert self.is_static_input_scheme is False
    # PATCH: skip Marlin repack for DSV4 wo_a/wo_b/compressor.wkv.
    # Those layers use the SM12 Triton fp8_einsum kernel directly,
    # which reads the original FP8 layout, not Marlin tiles.
    if getattr(layer, "is_bmm", False):
        if not hasattr(layer, "weight_scale") and hasattr(layer, "weight_scale_inv"):
            layer.weight_scale = layer.weight_scale_inv
        layer.input_scale = None
        return
    self.fp8_linear.process_weights_after_loading(layer)
    layer.input_scale = None
    return
```

See `patches/sm120_skip_marlin_for_bmm.diff` for the exact diff. Not yet
filed upstream — this is a structural design decision that needs a
maintainer-side discussion (the right long-term fix may be to give
`is_bmm=True` layers a different `quant_method` entirely, rather than
the same FP8_BLOCK scheme with a runtime branch).

## Memory budget

NVFP4-FP8-MTP is 172 GB on disk vs 159 GB for the W4A16 sibling. At TP=2
that's ~86 GB per rank loaded — close to the 96 GB HBM budget. After
loading, the cudagraph capture + KV cache squeeze tightly. To fit:

- `--gpu-memory-utilization 0.97` (up from W4A16's 0.95)
- `--max-num-seqs 8` (down from W4A16's 16)
- `--max-num-batched-tokens 2048` (down from W4A16's 8192)
- `--compilation-config '{"max_cudagraph_capture_size": 16, "cudagraph_capture_sizes": [1, 2, 4, 8, 16]}'`
  (reduced capture sweep; default captures 1..64 in 8 sizes)

These are TP=2 settings. TP=4 has more headroom (43 GB per rank model)
and can run with default cudagraph sizes if HBM permits.

## Quick repro

```bash
# One-time: build vLLM with TORCH_CUDA_ARCH_LIST=12.0a, apply the 7 patches
# (4 standard + 3 SM12-specific). See scripts/install_vllm_with_patches.sh
# for the universal patches; the 3 SM12-specific patches are documented
# in patches/sm120_*.diff in this repo.

# Download the artifact
hf download canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --local-dir /scratch/weights/nvfp4-fp8-mtp

# Serve at TP=2
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/nvfp4-fp8-mtp 8089 2

# Or TP=4
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/nvfp4-fp8-mtp 8089 4

# Smoke + benchmark
bash scripts/bench_rtx6000pro_suite.sh http://localhost:8089 2 1
```

## Caveats

- **MTP `num_speculative_tokens=1` only**: SM 12.0 attention paths
  assert `next_n==1` in the speculative-decode codepath. The B300
  recommendation (`num_speculative_tokens=2`) is not supported on this
  hardware.

- **`swiglu_limit` clamp may not be applied by Marlin MoE**: The vLLM
  selector excludes Marlin from `swiglu_limit` models specifically
  because Marlin doesn't apply the `±swiglu_limit` clamp on activations.
  We bypass this filter to get serving working at all. End-task
  benchmarks (GSM8K 50-prompt: 88% strict-match) look healthy, suggesting
  the missing clamp doesn't catastrophically hurt this model in
  practice — but if you care about high-precision math/reasoning
  workloads, evaluate end-to-end before deploying.

- **No-NVLink penalty**: At TP=4 (PCIe NCCL only) the per-replica
  throughput is bounded by all-reduce, not by compute. The B300
  measurements use NVLink and have a different balance.
