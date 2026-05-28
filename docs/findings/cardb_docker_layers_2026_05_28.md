# Card B Docker on RTX PRO 6000 — 12-layer dependency archaeology (2026-05-28)

This session reproduced the verified-on-host install_rtx6000pro_v3.sh recipe inside a clean Docker container, motivated by the HF community-#1 question asking for a prebuilt image. The result is a working Dockerfile (`docker/Dockerfile.rtx6000pro`) that gets through 11 layers of issues, but is currently blocked at layer 12 (attention `wo_a` projection loading as unquantized).

The point of this doc: a future operator (you, or someone trying to reproduce) doesn't have to rediscover layers 1-11 the hard way.

## Baseline environment

- Host: AWS g7e.24xlarge Brev `familiar-teal-worm`, 4× RTX PRO 6000 Blackwell Server Edition (96 GiB HBM, SM 12.0)
- Base image: `nvcr.io/nvidia/pytorch:26.04-py3` (torch 2.12.0a0+0291f960b6.nv26.04, Python 3.12.3, CUDA 12.9)
- Target: `jasl/vllm@27fd665bdc3ba58afc5c34cbb9034c9fc1a95029` on branch `ds4-sm120-preview-dev`
- Model: `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` (Card B)

## Layer-by-layer issues encountered

### Layer 1 — Docker data-root on /
**Symptom**: `failed to copy: write /var/lib/containerd/...: no space left on device` during base image pull.

**Root cause**: AWS DLAMI root partition is 248 GB with 11 GB free; Docker default data-root is `/var/lib/docker` and containerd content store is `/var/lib/containerd`.

**Fix**:
- Set `/etc/docker/daemon.json` `data-root` to `/opt/dlami/nvme/docker`
- Move containerd: `rsync /var/lib/containerd /opt/dlami/nvme/containerd; symlink /var/lib/containerd → /opt/dlami/nvme/containerd`

### Layer 2 — Missing setuptools_rust during vllm setup.py
**Symptom**: `ModuleNotFoundError: No module named 'setuptools_rust'` during `pip install -e . --no-build-isolation`.

**Root cause**: vllm's setup.py imports setuptools_rust unconditionally for the rust extension. NGC PyTorch 26.04 doesn't ship it.

**Fix**: Install `setuptools_scm setuptools_rust` before vllm pip install.

### Layer 3 — packaging<24.2 blocks setuptools 77+
**Symptom**: `ImportError: Cannot import packaging.licenses`.

**Root cause**: NGC PyTorch 26.04 pins `packaging==23.2` via `/etc/pip/constraint.txt` (the docker container ships a global pip constraint file). Modern setuptools needs packaging≥24.2 for `packaging.licenses`.

**Fix**: Use `PIP_CONSTRAINT=` (empty value) env var in RUN commands to override the constraint file.

### Layer 4 — NGC 25.04 base lacks torch/headeronly/
**Symptom**: `fatal error: torch/headeronly/util/Float8_e4m3fnuz.h: No such file or directory` during csrc compile.

**Root cause**: NGC PyTorch 25.04 ships torch 2.7.0a0 which doesn't have the `torch/headeronly/` namespace that vllm's `csrc/libtorch_stable/quantization/vectorization.cuh` needs. jasl/vllm's csrc references this header path.

**Fix**: Move to `nvcr.io/nvidia/pytorch:26.04-py3` (torch 2.12.0a0+0291f960b6.nv26.04).

### Layer 5 — torch ABI break from pip resolver
**Symptom**: `ImportError: vllm/_C.abi3.so: undefined symbol: _ZNR5torch7Library4_def...`.

**Root cause**: After installing `requirements/common.txt`, pip's resolver pulled PyPI torch 2.12.0 stable, replacing NGC's 2.12.0a0+0291f960b6 pre-release. vllm/_C was compiled against NGC's ABI; the upgrade broke it.

**Fix**: Generate a constraints file pinning the installed torch version, pass via `PIP_CONSTRAINT=/tmp/torch-pin.txt`:
```bash
echo "torch==$(python3 -c 'import torch; print(torch.__version__)')" > /tmp/torch-pin.txt
PIP_CONSTRAINT=/tmp/torch-pin.txt pip install -r requirements/common.txt
```

### Layer 6 — Debian PyYAML blocks pip upgrades
**Symptom**: `Cannot uninstall PyYAML 6.0.1 — The package's contents are unknown: no RECORD file was found for PyYAML`.

**Root cause**: NGC's image has `python3-yaml` installed via apt, which doesn't create a pip-readable RECORD file. pip can't uninstall what apt installed.

**Fix**: `apt-get remove -y --purge python3-yaml` before any pip install that pulls PyYAML.

### Layer 7 — `humming` module not importable
**Symptom**: `ModuleNotFoundError: No module named 'humming'` in `vllm/model_executor/layers/quantization/humming.py:49`.

**Root cause**: jasl's vllm quantization registry does `from .humming import HummingConfig` eagerly when ANY quantized model loads — not just humming-quantized ones. Card B uses compressed-tensors NVFP4 but the import happens anyway.

**Fix**: `pip install humming-kernels` (it's on PyPI).

### Layer 8 — flashinfer missing in workers
**Symptom**: `ModuleNotFoundError: No module named 'flashinfer'` in worker proc init.

**Root cause**: Workers import flashinfer for attention. The package is in jasl's `requirements/cuda.txt` with `[cu13]` extras pin (`flashinfer-python==0.6.11.post2`).

**Fix**: `pip install flashinfer-python flashinfer-cubin` (both on PyPI; cu12 path works fine).

### Layer 9 — split_module() doesn't accept tuple_return kwarg
**Symptom**: `RuntimeError: Worker failed with error 'split_module() got an unexpected keyword argument 'tuple_return''`.

**Root cause**: jasl's `vllm/compilation/backends.py:595` does:
```python
has_tuple_return = is_torch_equal_or_newer("2.12.0.dev")
tuple_return_kwarg = {"tuple_return": True} if has_tuple_return else {}
```
NGC's torch 2.12.0a0+0291f960b6.nv26.04 version-compares as ≥ 2.12.0.dev (correct), but the NGC torch fork doesn't have the `tuple_return` parameter on `torch.fx.passes.split_module.split_module` yet. PyPI torch 2.12.0 stable does.

**Fix**: Python heredoc patch (sed didn't take effect, likely due to Docker quote handling):
```python
sed equivalent — use python heredoc:
old = '        has_tuple_return = is_torch_equal_or_newer("2.12.0.dev")'
new = '        has_tuple_return = False  # canada-quant: NGC torch fork lacks tuple_return kwarg'
```

### Layer 10 — Inductor lowering on moe_forward_shared
**Symptom**: `LoweringException: AttributeError: Tried to call __getattr__ with attr 'stride' on a FakeScriptObject, implying that you are calling this inside of a fake kernel. ... target: vllm.moe_forward_shared.default`.

**Root cause**: The MoE forward custom op uses an opaque `torch.ScriptObject` as one of its args. Inductor's fake-tensor lowering tries to query `.stride()` on it during shape inference and fails. The hint suggests `register_opaque_type(members=...)` but jasl doesn't call it for this op.

**Fix (workaround)**: `--enforce-eager` flag on `vllm serve` to disable Inductor compilation entirely. Trade-off: no CUDA graph capture, conservative throughput. Not ideal but unblocks serving.

### Layer 11 — tilelang missing
**Symptom**: `RuntimeError: Worker failed with error 'tilelang is required for mhc but is not installed. Install it with pip install tilelang'`.

**Root cause**: jasl's mhc (Multi-Head Compressed) attention path JIT-compiles kernels via tilelang (`mhc_post_tilelang`, `hc_head_fuse_tilelang`). The package is in cuda.txt but not in our explicit install list.

**Fix**: `pip install tilelang` (on PyPI).

### Layer 12 (current blocker) — wo_a has no scale attribute
**Symptom**: `AttributeError: 'ColumnParallelLinear' object has no attribute 'weight_scale'` at `models/deepseek_v4/attention.py:365` during `determine_available_memory()` profile pass.

**Root cause**: `self.wo_a` (the attention output low-rank projection) loaded as a plain `ColumnParallelLinear` with no quantization wrapper. Neither `weight_scale_inv` (block-FP8 naming) nor `weight_scale` (regular FP8 naming) exists on the layer. Card B's safetensors header DOES have `wo_a.weight_scale_inv` for block-FP8 128×128 stored on disk — but the loader didn't wire it onto the layer object.

**Speculation**: VLLM_TEST_FORCE_FP8_MARLIN=1 may be short-circuiting the block-FP8 attention loader, treating wo_a as unquantized. Or jasl needs another upstream patch beyond PR #43723 that handles the layer construction itself (not just the scale lookup).

**Status**: Unresolved.

## Three real bugs in install_rtx6000pro_v3.sh found this session

These would have hit anyone running v3 on a fresh host (without prior vllm install state):

1. **Wrong patch path for PR #43723**: script targets `vllm/models/deepseek_v4/nvidia/ops/attention.py` — actual file at jasl@27fd665b is `vllm/models/deepseek_v4/attention.py` (no `nvidia/ops/` prefix). The patch silently doesn't apply.
2. **PR #40923 guard false-positive**: `grep -q "12.0a;12.1a" CMakeLists.txt` matches OTHER kernel arch lists in CMakeLists.txt (SCALED_MM, FP4, MLA all extend to 12.0a;12.1a), so the conditional thinks #40923 is applied when MARLIN_MOE_ARCHS is still `"8.0+PTX"` (silent JIT-PTX corruption on sm_120). Use a tighter grep: `grep -q 'cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX"' CMakeLists.txt`.
3. **`.deps/deepgemm-src` doesn't exist at this pin**: jasl@27fd665b doesn't have the bundled DeepGEMM under .deps; it has `vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` as a self-contained shim instead. v3's step 7 is a no-op.

## What "verified 2026-05-27" actually means

v3 install script header claims `Card B (NVFP4-FP8-MTP): serves coherent multi-step math (17*23 → 391 chain), MTP shared embeddings load, smoke completion clean`. This is a **single-completion smoke test**, not a full benchmark.

Card D (W4A16-FP8-MTP) on the same recipe DID get a full bench (24/30 AIME c=4, 91.6% MTP). So the recipe is more stable for Card D than Card B as of this writing.

## How far the Docker got before layer 12 blocked it

After layers 1-11 fixed, the container:
- ✅ Imports vllm 0.21.1rc1.dev363+g27fd665bd
- ✅ vllm CLI usable (`vllm --help` and `vllm serve` both work)
- ✅ All 4 worker ranks initialized over NCCL
- ✅ All 35 safetensors shards loaded into GPU memory (43 GB per GPU)
- ✅ TileLang JIT compiled `mhc_post_tilelang` + `hc_head_fuse_tilelang`
- ✅ Engine handshake past first collective_rpc
- ❌ Crashed in `determine_available_memory()` forward-pass profile at wo_a scale lookup

## Artifacts in this repo (this session)

- `docker/Dockerfile.rtx6000pro` — full Dockerfile (current state, blocked at layer 12)
- `scripts/bench_docker_full.sh` — 8-phase autonomous bench harness
- `scripts/aime_thinking_bench.py` — async AIME thinking-mode bench, compatible with existing JSON schema in `benchmarks/rtxpro6000/cardb_aime30_c*_thinking.json`
- `docs/findings/cardb_docker_layers_2026_05_28.md` — this doc

## Open question for next iteration

What needs to happen for `wo_a` to be constructed as a block-FP8 quantized layer with `weight_scale_inv` registered? Possibilities to investigate:
- Drop `VLLM_TEST_FORCE_FP8_MARLIN=1` and see if the natural NVFP4 oracle path picks up block-FP8 attention correctly
- Look for a jasl/vllm commit newer than 27fd665b that touches wo_a layer construction
- Patch the layer registration in `vllm/models/deepseek_v4/attention.py` to fall back to a synthesized unit scale (last-resort, may corrupt outputs)
- Build with the `quack-kernels` / `tokenspeed-mla` paths active (verify they're imported, not just installed)
