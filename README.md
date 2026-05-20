# dsv4-flash-nvfp4-fp8-mtp

Produces **the first DeepSeek-V4-Flash NVFP4-FP8 quantization that preserves the MTP speculative-decoding head.**

| | This repo | `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` |
|---|---|---|
| Quantization math | NVFP4 experts + FP8 attn | identical |
| Calibration corpus | ultrachat_200k 768@512 | same |
| **MTP layer** | **preserved** | **dropped** (transformers strips `mtp.*`) |
| vLLM `--speculative-config` | works | no draft model |
| Spec-decode throughput | ~1.5-2× on agentic | baseline only |

## How

RedHat's release uses stock `transformers.DeepseekV4PreTrainedModel`, which has `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` — silently drops every MTP weight at load time. This repo carries `patches/modeling_deepseek_v4.py.diff` (removes that filter) plus a vendored upstream `model.py` with init-skip + decoupled expert sharding, so MTP keys make it through calibration intact.

The quantization recipe (NVFP4 experts + FP8_BLOCK attn) is identical to RedHat's; the difference is purely architectural — what arrives at the quantizer.

### Verified against RedHat's published artifact (2026-05-20)

Direct inspection of `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` via the HF API:

- `model.safetensors.index.json`: **0 of 133,918 tensor keys match `mtp`** → MTP weights absent
- `config.json`: `num_nextn_predict_layers: 1` declared in architecture, but no MTP weights to back it
- Net effect for users: `vllm serve ... --speculative-config method=mtp num_speculative_tokens=2` against RedHat's artifact will fail to load the draft model

This repo's `verify_mtp_keys.py` confirms the inverse on our saved output (≥6 unconditional MTP Linear weights present + scales).

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Bootstrap (B300, venv-calib, venv-serve, /scratch/weights/bf16-mtp) | done (reuses sibling repo's assets) |
| 1 | BF16 dequant with MTP keys preserved | done |
| 2a | Multi-rank NVFP4 dryrun (gate test for B300 NCCL bug applicability) | **next** |
| 2b | 8-rank NVFP4 full calibration | gated on 2a |
| 3 | Post-process for vLLM | scaffolded |
| 4 | Verify MTP retention | scripts ready |
| 5 | vLLM serve smoke (--speculative-config method=mtp) | TBD |
| 6 | Bench vs RedHat + MTP differential | TBD |
| 7 | HF model card + release | TBD |

See [PLAN.md](PLAN.md) for the full plan. See [CLAUDE.md](CLAUDE.md) if you're a Claude Code agent resuming work here.

## Sibling repo

The W4A16-GPTQ recipe (different math, different artifact) lives at `canada-quant/dsv4-flash-w4a16-fp8-mtp`. NVFP4 was chosen for this repo because:
- The B300 NCCL bug we hit on the W4A16 path is in `GPTQModifier._reduce_hessian_to_target_rank` — GPTQ-specific
- `QuantizationModifier` (used for NVFP4) has zero `dist.*` calls in its main file, so multi-rank should work
- NVFP4 has hardware tensor-core support on Blackwell B300 — right format for the hardware
- RTN-style calibration is 5-10× faster than GPTQ; a single 8-rank run completes in 4-12h

## License

Apache-2.0 (matches DeepSeek's release license).
