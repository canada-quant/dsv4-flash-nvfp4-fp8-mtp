# Upstream research — DSV4 / NVFP4 / llm-compressor / compressed-tensors / vLLM

**Date:** 2026-05-20
**Triggered by:** standing rule "search upstream before working — every time." Refresh weekly during artifact work, and on every session resume.

## TL;DR

- **llm-compressor `vllm-project/llm-compressor#2734` already exists** — filed by `pasta-paul` (sibling H200 agent) today at 19:44 UTC for the GPTQ-Hessian variant of the multi-rank disjoint-module-set hang. It explicitly names our `Observer.synchronize` patch as "patch A defensive." We need to **comment** on this issue with the NVFP4 weight-only confirmation, not file a new issue.
- **vLLM `vllm-project/vllm#42209` (sychen52) adds NVFP4 MOE for DSV4 — `ready` label, very active.** Our artifact's vLLM serve path depends on this PR merging or us pulling the branch.
- **vLLM `vllm-project/vllm#41276` (kylesayrs)** is the umbrella WIP DSV4 quantization support PR — body documents the RedHat NVFP4-FP8 artifact deployment command (`vllm serve RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8 --tensor-parallel-size 4 --port 8089 --kv_cache_dtype=fp8`) and GSM8K 0.910 baseline. This is the reference deployment we benchmark against.
- **llm-compressor `vllm-project/llm-compressor#2647` (kylesayrs/transformers-v5) is the branch that produced the RedHat NVFP4-FP8 artifact.** OPEN, `needs-rebase` label, updated 1h ago. Our `/data/venv-calib` is pinned to `f2aa32e2` from this branch.
- **vLLM bool() one-liner: bug exists on 5 sites, not 2.** Confirmed against `vllm-project/vllm` `main` head. Sites: lines 621, 628, 642, 650, 673 in `compressed_tensors.py`. Our PR scope needs adjustment.
- **Org rename note:** `llm-compressor` is now under `vllm-project/`, not `neuralmagic/`. CLAUDE.md PR queue needs correction.

## Key PRs / issues by repo

### `vllm-project/vllm`

| # | State | Author | Title | Why it matters |
|---|---|---|---|---|
| #41511 | open issue | us (pasta-paul) | compressed-tensors W4A16 MoE: weight_scale not sharded along K under TP | Our prior contact w/ kylesayrs / vLLM team. Still open. |
| #40902 | open issue | vLLM team | [Roadmap] DeepSeek V4 | Tracks the broader DSV4 work; subscribe / refresh. |
| #41276 | open PR | **kylesayrs** | [WIP] [DSV4] Quantization Support | Umbrella; body documents RedHat NVFP4-FP8 deployment + GSM8K 0.910. Reference for our model card. |
| #41533 | closed/merged | vLLM team | [DSV4] Attention accumulation in model dtype | Pre-req merged. |
| #42209 | **open PR (ready)** | sychen52 | Add NVFP4 MOE support for Deepseek V4 | **Required for our artifact to serve.** Branch `sychen52:nvfp4_dsv4` at `af11a8480c4`. Reviewers: tlrmchlsmth, mgoin, kylesayrs's team, Harry-Chen, WoosukKwon, robertgshaw2-redhat, xinli-sw. Very active, last update 2026-05-20 19:50 UTC. |
| #42444 | closed/merged | vLLM team | [Model Runner V2][Bug Fix][DSV4] Ensure lazy attention state initializations happen during cudagraph capture | Recent fix may affect serve smoke. |
| #42562 | open PR | DSv4 team | [Perf][DSv4] Add cuteDSL generic LL router GEMM | Performance, not blocking. |
| #42970 | **open PR** | vLLM team | [Bugfix][DeepSeek V4] Resolve expert_dtype for FP8 checkpoints missing the field | **Affects our config.json** — we set `expert_dtype: fp4`. If RedHat's lacks the field, vLLM may treat ours and theirs differently. Confirm behavior. |
| #42516 | open issue | community | Gemma4 NVFP4 fails to start with PP=2, or TP=2 without EP | Same class as our concern; may indicate broader NVFP4 + parallelism bugs. |
| #43009 | open issue | community | Triton kernel JIT compilation during inference | Not directly relevant but watch for B300 implications. |

### `vllm-project/llm-compressor`

| # | State | Author | Title | Why it matters |
|---|---|---|---|---|
| #2647 | **open PR (needs-rebase)** | **kylesayrs** | Transformers v5 (head `kylesayrs/transformers-v5`) | The branch that produced the RedHat NVFP4-FP8 artifact. Our `/data/venv-calib` is pinned to `f2aa32e2` from this branch. |
| #2734 | **open issue** | **us (pasta-paul)** | GPTQModifier hangs on multi-rank with sharded MoE experts | **Filed today 19:44 UTC.** Covers GPTQ-Hessian; explicitly names Observer.synchronize as "patch A defensive." We comment, not duplicate. |

### `vllm-project/compressed-tensors`

12 results for `dispatch_with_map OR from_accelerate OR "is not an nn.Module"`. None match our specific sharded-MoE crash. Closest:

| # | State | Author | Title | Why it matters |
|---|---|---|---|---|
| #647 | open issue | kylesayrs | [Performance] Speed up model loading by desynchronizing sync ops in `from_accelerate` | **Perf issue**, not our bug. Same code region. Could be useful context when filing ours. |
| #688 | open PR | kylesayrs | feat: validate shared memory segment limits before CPU offloading | Adjacent offload work. |
| #676 | closed PR | Etelis | fix: skip double offload when from_pretrained wrapper runs twice | Closed; the offload-twice path is what we already catch in our oneshot try/except. |
| #626 | closed PR | kylesayrs | [Offload] Make sure disk operations only happen on rank 0 | Establishes the rank-0-only save pattern we had to break for sharded-MoE. |
| #698 | open PR | dichn | [Offload] use weakref.finalize to handle shared tensor deletion | Offload bookkeeping. |
| #704 | closed PR | HDCharles | fixing some issues with @torchrun decorator | Multi-rank test infra; useful if our PR needs a multi-rank test. |

**Our dispatch_with_map AttributeError on sharded modules is genuinely unreported.** We file a fresh issue.

### `jasl/vllm-ds4-sm120-harness`

Cloned to `/data/harness/` on box. Recent activity research deferred to harness-prep tier (concurrent with calibration). Key files identified during README scan:
- `configs/sm120_tp2_serve.env.example` and `configs/gb10_sm121_serve.env.example` — NOT B300-applicable, do not copy verbatim.
- `baselines/20260502_b200_tp4_main_5737770c6/` — B200 TP=4 baseline bundle WITH `nomtp` and `mtp` reference data. Usable for behavior-compare regardless of our SM arch.
- `tests/test_official_baseline.py`, `tests/test_oracle.py`, `tests/test_generation_alignment.py` — server-agnostic correctness tests.

## Maintainer-response status (load-bearing for action ordering)

| Issue/PR | Repo | Status | Last maintainer touch |
|---|---|---|---|
| #41511 (W4A16 MoE TP sharding) | vllm-project/vllm | open | **none — zero maintainer comments since 2026-05-04** |
| #2734 (multi-rank sharded MoE hang) | vllm-project/llm-compressor | open | **none — filed today 2026-05-20 by sibling, zero comments** |

Both our prior work issues are still awaiting maintainer attention. Our action: surface the artifact reproducer in the queued PRs to give the issues additional weight rather than file more dangling reports.

### `jasl/vllm-ds4-sm120-harness`

Harness already cloned to `/data/harness/` on the box. Targets SM120/SM121 on the build/profile side but is hardware-agnostic on the **client** side. Has first-class MTP variant support, B200 baselines, and runtime MTP acceptance-rate telemetry (the headline differentiator metric for our artifact). See `PLAN.md` Harness section.

## Implications for the artifact work

### 1. Stop assuming "we file the upstream PR"

`pasta-paul/canada-quant` is already in the upstream relationship with the sibling repo's issue #2734. Action shifts:

- **llm-compressor:** comment on #2734 with NVFP4-specific reproducer (observer-sync fires even on weight-only RTN recipes because `match_named_modules` is per-module not per-Hessian). Offer to follow up with the PR using the "replication group" attribute design proposed in #2734's body. Tag `@kylesayrs`.
- **vLLM bool() one-liner:** still file fresh. Confirmed bug on 5 sites (lines 621, 628, 642, 650, 673 of `compressed_tensors.py`). The sibling repo had 2 sites in its memory entry — that count was based on the W4A16 commit; main has more now.
- **compressed-tensors `dispatch_with_map`:** research pending. May already be reported.

### 2. The vLLM serve path is gated on PR #42209 merging

Our artifact cannot be served via stock vLLM main today. Options:

- **Wait for #42209 merge** (`ready` label, active reviews — likely soon).
- **Pull #42209's branch into `/data/venv-serve`** for immediate testing. Branch `sychen52:nvfp4_dsv4` at `af11a8480c4` rebases onto vLLM main.
- **Comment on #42209** with our artifact as additional real-world reproducer; this builds the upstream relationship and gives the reviewers a non-RedHat NVFP4 model to validate against.

### 3. The model card has a concrete baseline to beat

Kyle's PR #41276 documents the RedHat artifact at **GSM8K 0.910**. Our artifact's GSM8K must hit ≥ 0.91 (parity) AND report a non-zero MTP acceptance rate (which RedHat literally cannot). This is the headline comparison.

Deployment command (from #41276 body):
```bash
vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --tensor-parallel-size 4 --port 8089 \
    --kv_cache_dtype="fp8" \
    --speculative_config '{"method":"mtp","num_speculative_tokens":2}'
```

The `--speculative_config` is the additive piece RedHat's deployment doesn't include because they can't use it.

### 4. expert_dtype handling (PR #42970)

Our `_CompatConfig` sets `expert_dtype = "fp4"` in `config.json`. PR #42970 fixes the case where FP8 checkpoints are MISSING this field. Confirm whether our setting it explicitly to "fp4" plays well with the fix, OR whether the bugfix logic preempts our value. If it does, we may need to set it to a different sentinel.

## Action items (added to TaskList)

- **#8** updated: comment on llm-compressor #2734 (was: file new issue).
- **#15** new: rebase `/data/venv-serve` against vLLM PR #42209 OR plan to wait for merge.
- **#16** new: ongoing upstream activity watch (refresh weekly during artifact work).
- **#9** vLLM bool() PR scope corrected: 5 sites not 2.

## Files / branches to inspect next

- `vllm-project/llm-compressor` PR #2734 body (already inspected, captured above).
- Sibling repo's `scripts/multirank_patches.py` (referenced in #2734 body) — canonical patch implementation for the replication-group fix design.
- vLLM PR #42209 actual diff — what code paths it touches, whether it affects MTP loading.
- vLLM PR #42970 actual diff — what `expert_dtype` resolution logic looks like.
- `vllm-project/compressed-tensors` search for `dispatch_with_map`, `from_accelerate`, `sharded` — pending in next batch.
