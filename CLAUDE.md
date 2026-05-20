# CLAUDE.md — session notes for Claude Code agents

If you're a Claude Code agent resuming work in this repo, **read this first** along with [`PLAN.md`](PLAN.md). The user's persistent memory at `~/.claude/projects/-home-paul/memory/MEMORY.md` carries cross-project context.

## Quick context

This repo produces the **first** DeepSeek-V4-Flash NVFP4-FP8 quantization that **preserves the MTP layer**. RedHat shipped `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` but they used stock HF transformers' `DeepseekV4PreTrainedModel`, which has `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` and silently drops the MTP speculative-decoding head at load time. Our edge: we carry the modeling-class patch + vendored upstream model + decoupled expert sharding from sibling repo `canada-quant/dsv4-flash-w4a16-fp8-mtp`, and we use **`QuantizationModifier` (RTN-style), not `GPTQModifier`**, which sidesteps the B300 NCCL+NVLink hang we hit on the GPTQ path.

**Topology decisions:**
- **Quantization scheme:** `NVFP4` on routed MoE experts (group_size=16, tensor_group, FP8 e4m3 scales). `FP8_BLOCK` 128×128 on attention Linears + MTP `e_proj`/`h_proj`. Recipe matches RedHat's V4-Flash NVFP4-FP8 *exactly* on the math; the only delta is MTP retention.
- **Hardware:** Blackwell B300 (SM 10.0a). NVFP4 has hardware tensor-core support. This is the right box for this format.
- **Distributed:** multi-rank torchrun is expected to work because `QuantizationModifier` has zero `dist.*` calls in its main file (verified). NCCL hangs we saw on `GPTQModifier._reduce_hessian_to_target_rank` are GPTQ-specific. **STILL VERIFY** with a 4-rank dryrun before launching full 8-rank (observer code or compressed_tensors internals could still have collectives).

## Why a new repo

The sibling repo `canada-quant/dsv4-flash-w4a16-fp8-mtp` carries the GPTQ-path scaffolding, 7 dryrun friction patches, and is sized for the *predecessor's* W4A16-FP8 recipe. It's the right home for any GPTQ work (now happening on the predecessor H200 box per the active strategy in that repo's `CLAUDE.md`). NVFP4 is a fundamentally different code path (different `Modifier`, different scheme, no Hessian, different vLLM serve backend), gets a different model card, and ships from a different org (`canada-quant`). Keeping it separate avoids the "what does THIS repo actually produce?" confusion that hits multi-recipe repos.

## Hardware + AWS

- **EC2:** `i-0714f36a266c8c59b`, `p6-b300.48xlarge`, `us-west-2`. **Profile `rozo`** (not default). 3-day prepaid spot reservation.
- **SSH:** `ssh -i ~/.ssh/qwenv4-quant.pem ubuntu@35.161.108.205`
- **DLAMI:** Ubuntu 24.04 + `/opt/pytorch` Python 3.13 venv with torch 2.11.0+cu130. CUDA 13 bundled at `/opt/pytorch/cuda` is runtime-only; for source builds (vLLM, FlashAttention) use `/usr/local/cuda` after `sudo apt install cuda-toolkit-13-0`.
- **Existing assets on box** (from the sibling repo's work, can be reused):
  - `/data/venv-calib` — pyenv with torch 2.11.0+cu130, compressed-tensors `0.15.1a20260515`, llmcompressor `f2aa32e2`, py-spy 0.4.2
  - `/data/venv-serve` — pyenv with vLLM `0.1.dev1+g3424fba51.d20260519` (jasl/dm120 + Blackwell MTP fixes)
  - `/scratch/weights/bf16-mtp/` — the dequantized BF16 source with MTP keys intact (568 GB across 46 safetensors)
  - `/data/vendor/dsv4-upstream/` — upstream's verbatim `model.py`, `kernel.py`, `config.json` (NVFP4 path uses the same vendored model)

## Critical gotchas

1. **`mtp.*` silent drop** — transformers 5.8.1 (latest pypi) has `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` on `DeepseekV4PreTrainedModel`. `patches/modeling_deepseek_v4.py.diff` hunk 1 removes it. STILL CURRENT on `transformers` main branch as of 2026-05-20.
2. **B300 NCCL+NVLink hang on GPTQ Hessian reduce** — `GPTQModifier._reduce_hessian_to_target_rank` deadlocks on multi-rank B300 (any of 2/4/8 ranks). Symptom: workers stuck at `compress_module_list:304` indefinitely with `cudaStreamSynchronize`, NVRM "Failed to send inband data" in dmesg. Verified workarounds tried: NCCL_P2P_DISABLE=1, NCCL_NVLS_ENABLE=0, NCCL_DEBUG=INFO. None helped. **NVFP4 uses `QuantizationModifier` instead which has NO `dist.*` calls — should be immune. VERIFY WITH DRYRUN.**
3. **Decoupled expert sharding** — `scripts/upstream/__init__.py` implements `_expert_world_size`-based MoE sharding (keeps upstream's `world_size=1` for embed/head/attn since `Column/RowParallelLinear` are rebound to plain Linear, but shards experts N-way independently). Loadtest validated: 26 GB per rank on 8 ranks (vs 568 GB unsharded). Required for any multi-rank V4-Flash work on a box with <5 TB system RAM.
4. **Don't use `--system-site-packages`** — `/opt/pytorch`'s Python 3.13 venv inheriting `/usr/lib/python3/dist-packages` (3.12-compiled wheels) crashes with `pyo3_runtime.PanicException` on `cryptography` import.
5. **vLLM serve recipe for NVFP4** — vLLM picks `compressed_tensors_moe_w4a4_nvfp4` backend for `nvfp4-pack-quantized` format. Different code path from `WNA16Marlin` used for W4A16. The serve-side config fixes from the sibling repo's `postprocess_for_vllm.py` mostly transfer, but the targets regex and scale handling may differ — verify in serve smoke before claiming compatibility.

## Working norms

- The user prefers terse, factual responses. Don't write trailing summaries unless asked.
- Confirm before risky/expensive actions (HF upload, force-push, instance termination).
- Commit messages: include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (heredoc multi-line, not `-m` flags).
- Org: `canada-quant` (GitHub) + Canada Quant Labs (planned HF org).
- The repo is **PRIVATE** until the user authorizes Phase 8 (HF release).

## What's where in the repo

```
PLAN.md                          — phase-by-phase plan, recipe, risk register
patches/
  modeling_deepseek_v4.py.diff   — transformers patch: remove _keys_to_ignore_on_load_unexpected for mtp
  helpers.py.diff                — llm-compressor patch: dotless tensor match fix
  VERSIONS.md                    — pinned versions of every patched dependency
scripts/
  bootstrap_p6_b300.sh           — idempotent box setup (apt, venvs, patches)
  quantize_v4_nvfp4_fp8_mtp.py   — Phase 2 calibration entry point (NVFP4, RTN-style, multi-rank)
  load_bf16_into_transformer.py  — BF16 -> Transformer loader with rename rules + alias handling
  calibration_model.py           — CalibrationModel wrapper that drives main layers + MTP forward
  loadtest_sharded.py            — N-rank load test (RSS sanity before full run)
  postprocess_for_vllm.py        — transforms saved artifact to vLLM-loadable form
  verify_mtp_keys.py             — confirm MTP keys present in saved artifact
  verify_mtp_quantized.py        — confirm MTP layer's weights are actually quantized (not bf16 pass-through)
  upstream/
    __init__.py                  — vendored-model shim: GPTQLinear init-skip, decoupled MoE shard,
                                   dist-mask around Transformer construction
    kernel_shim.py               — stubs tilelang kernels (sparse_attn, hc_split_sinkhorn) for calibration
vendor/dsv4-upstream/
  model.py                       — DeepSeek's inference/model.py verbatim (the patch target)
  kernel.py                      — DeepSeek's kernel.py (only imported via shim)
  config.json                    — upstream config for ModelArgs reconstruction
notes/                           — design docs, debugging session notes (gitignored from main flow)
```

## Resuming a session

1. SSH the box. Verify `/data/venv-calib` exists: `ls /data/venv-calib/bin/python`. If the instance was stopped, `/data` persists but `/scratch` is wiped — re-stage weights from S3 if so.
2. Check `/scratch/weights/bf16-mtp/` exists. If not, the sibling repo's `scripts/bootstrap_p6_b300.sh` shows how it was produced.
3. Read `PLAN.md` for the current phase.
4. First action: a 4-rank dryrun of NVFP4 (`scripts/quantize_v4_nvfp4_fp8_mtp.py --dry-run-one-layer --samples 16`) to confirm `QuantizationModifier` doesn't hit the GPTQ-path NCCL bug on B300. Only after dryrun is clean: launch 8-rank full run.

## How to talk to this codebase

When a user says "the artifact" they mean **this repo's NVFP4+FP8+MTP output**, not RedHat's NVFP4-FP8 (no MTP) or the sibling repo's W4A16+FP8+MTP. "MTP" means the speculative-decoding head at upstream key prefix `mtp.0.*` (counter is 0-indexed within MTP, not 43 like the layer number in the architecture). "NVFP4" is the Blackwell-native FP4 format with FP8-e4m3 block scales. "Predecessor" still refers to `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` (the AWQ recipe shipped by the same person who owns the sibling W4A16-MTP repo); RedHat is a separate reference point.

## Differentiator vs RedHat

| Aspect | `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` | This repo's output |
|---|---|---|
| Quantization math | NVFP4 experts + FP8 attn | identical |
| Calibration data | ultrachat_200k 768@512 | same |
| MTP layer | **dropped** (transformers strips `mtp.*`) | **preserved** (our patch) |
| vLLM `--speculative-config` works | no | yes |
| Spec-decode tok/s gain | 0 (no draft model) | ~1.5-2× on agentic workloads |
| Differentiator durability | n/a | until RedHat forks transformers (no indication they will) |

The MTP retention is the entire reason to publish this. Make sure every shipping artifact (model card, README, HF metadata) leads with that.
