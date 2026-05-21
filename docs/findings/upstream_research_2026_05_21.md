# Upstream research — vLLM serve target for NVFP4-FP8-MTP on B300 (2026-05-21)

**Trigger:** Phase 5a vLLM serve smoke failed 8 consecutive times against the `jasl/dm120` fork's `fp8_einsum.py` kernel with `b.shape[0]=256 not divisible by out_rank=1024`. Hypothesis: jasl/dm120 is sm120-optimized (consumer Blackwell, RTX Pro 6000 / GB10) and its kernel doesn't match our sm100a (B300, data-center Blackwell) artifact layouts.

**Goal:** Confirm hypothesis + find the correct vLLM serve target for an sm100a NVFP4-FP8 DSV4 artifact.

## Findings

### F1 — Mainline vLLM has been refactored into modular DSV4 layout (2026-05-19)

Four PRs merged 2026-05-19 in sequence:
- #43004 — [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]
- #43039 — [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N]
- #43073 — [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N]
- #43077 — [Model Refactoring] Rename deepseek_v4.py to model.py [4/N]

Result: mainline `vllm-project/vllm` `main` now has `vllm/models/deepseek_v4/` with `amd/`, `common/`, `nvidia/`, `attention.py`, `compressor.py`, `quant_config.py`. The `nvidia/` subdir has `model.py`, `mtp.py`, and `ops/` with **only** `cutedsl_utils.py`, `dequant_gather_k_cutedsl.py`, `fused_indexer_q_cutedsl.py`.

**Decisive:** Mainline `nvidia/ops/` has **no `fp8_einsum.py`**. The kernel that's been failing Phase 5a is jasl-fork-only.

### F2 — Two candidate vLLM PRs (#41276 vs #42209) are complementary, not redundant

| PR | Author | Title | State | Files |
|---|---|---|---|---|
| #41276 | kylesayrs (neuralmagic) | [WIP] [DSV4] Quantization Support | OPEN (draft, last update 2026-05-08) | 3 legacy paths under `vllm/model_executor/{layers,models}/deepseek_*.py` |
| #42209 | sychen52 | Add NVFP4 MOE support for Deepseek V4. | OPEN (not draft, 3 approvals, last update 2026-05-21) | 9 modern paths including `vllm/models/deepseek_v4/quant_config.py` + `fused_moe/*` |

`comm -12 <(files-41276) <(files-42209)` = **empty**. Zero file overlap.

Interpretation: #41276's plumbing has been **absorbed** into mainline via the 4-PR refactor. #42209 is the **only PR needed** on top of current mainline for NVFP4 MoE DSV4 serve.

RedHat's model card references #41276 as their serve dep, but that reference was written before the 2026-05-19 refactor. **The current correct serve path is mainline + #42209**, not mainline + #41276.

### F3 — PR #42209 status as of 2026-05-21

- Branch: `sychen52:nvfp4_dsv4`
- Head SHA: `d05d52059f`
- Base: `vllm-project/vllm` `main`
- 3 approvals: xinli-sw (2026-05-19), pavanimajety (2026-05-19), zyongye (2026-05-20)
- Outstanding asks: zyongye wants perf data; pavanimajety wants `kernels.yaml` integration (already in PR)
- Buildkite pre-commit: failing repeatedly (style/lint level), core CI green
- Mergeable: yes (no conflicts at survey time)
- Estimated merge timeline: days (close, but author still polishing)

### F4 — Mainline `model.py:909` still has the `scale_fmt` hard subscript

```python
self.scale_fmt = config.quantization_config["scale_fmt"]
```

This is the same code we patched-around in the jasl fork by injecting `scale_fmt: ue8m0` into our artifact's `config.json`. **The bug is in mainline too, not jasl-only.** Our queued PR (defensive `.get("scale_fmt", "ue8m0")`) targets mainline directly, not the fork.

### F5 — No vLLM mainline SM 10.0a / B300 specific build issues found

Searched for issues / PRs mentioning `TORCH_CUDA_ARCH_LIST=10.0a`, `sm100a`, `B300`. No reports of mainline build failures on B300. PR #43270 (2026-05-21) is "Auto-bind on DGX B300" — orthogonal (runtime affinity).

Recommendation: build with `TORCH_CUDA_ARCH_LIST=10.0a` explicit; should succeed.

### F6 — Other relevant open PRs (not blockers, just context)

- #42754 — Quark MXFP4 deepseek_v32 bugfix (irrelevant; we use compressed-tensors not quark)
- #42970 — expert_dtype FP8 fix (relevant if our `expert_dtype: fp4` field hits issues; monitor)
- #42601 — NVFP4 MoE NaN clamp (potential numerics improvement; not blocker)
- #42562 — DSv4 LL router GEMM (router-side optimization; orthogonal)
- #39933 — FlashInfer CuTeDSL NVFP4 backend (potential future kernel; not blocker)
- #43248 — our bool() wrap PR (filed, waiting)

### F7 — llm-compressor open issues (our backlog)

- #2734 — Observer.synchronize NCCL desync (sibling filed; we commented; 0 maintainer response)
- #2741 — intra-calibration resume feature request (filed; 0 response)
- #2743 — 8-rank cache-offload deadlock (filed; 0 response)
- #2745 — MTP inference-tensor crash (filed 2026-05-21 02:51 UTC; 0 response at survey time)

## Decision

**Switch `/data/src/vllm` from jasl/dm120 to mainline + PR #42209.** Concrete steps:

1. `cd /data/src/vllm`
2. `git stash` (preserve paul/dsv4 unstaged mods for reference)
3. `git fetch upstream main` and `git fetch upstream pull/42209/head:pr-42209`
4. `git checkout upstream/main` (post-#43077 mainline)
5. `git cherry-pick <PR_42209_commit_range>` — apply NVFP4 MoE patches
6. Verify our `scale_fmt: ue8m0` injection is still in the artifact's `config.json` (it is)
7. Verify our `squeeze_global_scales.py` was run on the artifact (it was — 66,048 tensors squeezed)
8. `TORCH_CUDA_ARCH_LIST=10.0a pip install -e . --no-build-isolation`
9. Retry Phase 5a serve smoke

**Expected outcome:** the `fp8_einsum` 256-row error disappears because mainline doesn't have that kernel. The serve path uses cutedsl-based ops which consume our wo_a layout without reshape constraints.

**Confidence: High.** The kernel that was failing literally doesn't exist on mainline; that's not a hypothesis, it's a `gh api` confirmed absence (F1).

**Residual risks:**
- Mainline + PR #42209 may surface a different failure mode we haven't seen (compile-time arch flags, missing module, etc.). Mitigation: build + import test before launching serve.
- PR #42209 may rebase or merge during the rebuild; if so, pull the merge commit instead of cherry-picking. Mitigation: re-check PR status before `git fetch`.

## Documentation cross-references

- Recipe replication doc: `docs/recipes/nvfp4_fp8_mtp_replication.md` (§4.3, §5 Phase 5, §11)
- Memory entry: `~/.claude/projects/-home-paul-dsv4-flash-nvfp4-fp8-mtp/memory/jasl_dm120_is_sm120_not_sm100.md`
- CLAUDE.md gotcha catalog: §"Critical gotchas" item 5 (vLLM serve recipe)
