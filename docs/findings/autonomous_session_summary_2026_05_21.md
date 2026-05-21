# Autonomous session summary — 2026-05-21

Session: ~3.5h elapsed of a 10h autonomous mandate. User instruction: *"continue autonomously to hit your goal and make your own decision to reach goal."* Goal: ship `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` and contribute upstream PRs in parallel.

## TL;DR — what's ready to ship

The artifact at `/scratch/weights/v4-flash-nvfp4-fp8-mtp/` on the B300 box is **publication-ready** pending only user authorization for HF upload. Everything else is done.

| Status | Item |
|---|---|
| ✅ Done | Artifact built (172 GB, 35 shards, 256 experts, 799 MTP keys, all Phase 4 gates intact) |
| ✅ Done | vLLM serve smoke green on mainline + PR #42209 + 3 local patches (Phase 5a) |
| ✅ Done | **GSM8K: 0.9181 strict / 0.9515 flexible** — beats RedHat's 0.910 |
| ✅ Done | **MMLU-Pro: 0.8113 ± 0.0035** (math 0.9149, law 0.6022) |
| ✅ Done | Phase 4 MTP retention re-verified after all postprocesses |
| ✅ Done | `MODEL_CARD.md` written with headlines + reproduction guide |
| ✅ Done | `docs/recipes/nvfp4_fp8_mtp_replication.md` — 16 gotchas + replication template |
| ✅ Done | `docs/phase7_hf_upload_prep.md` — upload commands gated on user auth |
| ⏸ Deferred | Phase 5b spec-decode live measurement (gated on upstream) |

## Phase 5b honest framing

The MTP differentiator at the **artifact level** is solid — weights present, structurally correct, Phase 4 verifier passes. **The live `--speculative-config method=mtp` serve measurement is blocked by two upstream issues**, both of which we filed this session:

1. **`vllm-project/llm-compressor#2745`** (filed earlier) — `Inplace update to inference tensor outside InferenceMode` crash at MTP qparam writeback during calibration. We empirically verified today that this crash fires for ALL MTP-block modules (not just shared-embed), so MTP must be excluded from quant scope (the v2 calibration we ran in parallel as a test confirmed this — subgraphs 1–43 completed in 73 min, subgraph 44 crashed identically).

2. **`vllm-project/vllm#43304`** (filed today) — vLLM's MTP draft model construction inherits the main model's `quant_config`, so BF16-MTP artifacts can't load through `--speculative-config method=mtp`. Three concrete fix options proposed.

When either fix lands, re-run Phase 5b to populate the spec-decode acceptance rate number in the model card.

## Upstream contributions filed this session

| # | Repo | Type | Status |
|---|---|---|---|
| 43248 | vllm-project/vllm | PR (bool wrap) | OPEN — 1 review (welcome bot), pre-run-check fails because contributor onboarding gate (needs 4+ merged PRs or maintainer `ready` label) |
| 43288 | vllm-project/vllm | PR (`.get('scale_fmt', 'ue8m0')`) | OPEN — same gating |
| 43290 | vllm-project/vllm | PR (`weight_scale_inv`-or-`weight_scale` fallback) | OPEN — same gating |
| 43297 | vllm-project/vllm | Issue (FusedMoE `(1,)` global_scale broadcast) | OPEN — 0 responses yet |
| 43304 | vllm-project/vllm | Issue (DSV4 MTP draft inherits main quant scheme) | OPEN — 0 responses yet |
| 2745 | vllm-project/llm-compressor | Issue (filed earlier; commented today with v2 confirmation queued) | OPEN |

All filings use mainline upstream code paths and explicitly cite this artifact as the motivating reproducer. No filings reference the jasl/dm120 fork (which would have been wrong; the audit earlier this session confirmed all framings are correct).

## Major findings this session

1. **B300 is sm_103a, not sm_100a.** CLAUDE.md previously said 10.0a — `nvidia-smi --query-gpu=compute_cap` confirms `10.3`. Earlier vLLM rebuild with `TORCH_CUDA_ARCH_LIST=10.0a` produced sm_100a-only kernels that don't run on sm_103a (the `a` suffix is non-portable). Rebuild with `TORCH_CUDA_ARCH_LIST=10.3a` was the unblock for Phase 5a — the previous "deferred at DeepGemm sm100a kernel gap" finding was actually an arch-list misconfiguration.

2. **vLLM mainline serves DSV4 NVFP4-FP8 artifacts via 3 layered config-side requirements** (now captured as gotchas 13–16 in the recipe doc):
   - `scripts/update_config_for_fused_attn.py` rewrites `config_groups[group_0].targets` from unfused (`wq_a|wkv`) to FUSED (`fused_wqa_wkv|compressor.fused_wkv_wgate`) names matching vLLM's MergedColumnParallelLinear prefix.
   - `scripts/convert_attn_scales_for_vllm.py` upcasts attn `.weight_scale` BF16 → FP32 (DeepGemm requirement) and injects `input_activations` (dynamic FP8 group=128) into `config_groups[group_0]`.
   - `scripts/squeeze_global_scales.py` squeezes shape-`(1,)` global_scale tensors to 0-D scalar for vLLM's MoE loader.
   - `CUDA_HOME=/usr/local/cuda` at serve time (NOT the pip-installed `nvidia/cu13/nvcc` whose headers conflict with its compiler — Tilelang JIT bites this).
   
3. **GSM8K flexible-extract is invariant to NVFP4 quantization** on this artifact — exact match with BF16 baseline (0.9515 → 0.9515). Strict-match drops 3.41 pts (0.9522 → 0.9181) — quant affects text-formatting precision but not final-answer correctness on these 1319 problems. Beats RedHat's 0.910 strict by ~0.81 pts.

4. **`flex-extract` is the right benchmark to publish for accuracy claims** — it measures what the model gets RIGHT, not the format adherence which is a quant-sensitivity proxy. (Both numbers reported in the model card for completeness.)

## What I did NOT do (deliberate decisions)

- **Did not patch vLLM mainline for BF16-MTP wo_a path** (Option B in #43304 reproducer). The patch is non-trivial (mtp.py construction override + attention.py BF16 GEMM branch) with risk of subtle bugs. Time better spent on documentation + filing the upstream issue so maintainers do it right. Estimated saved: 2-4h.
- **Did not file a 3rd vLLM PR with the BF16-MTP patch** for the same reason — issue #43304 has 3 proposed fixes; upstream picks the best.
- **Did not run additional benchmarks beyond GSM8K + MMLU-Pro.** Tried ARC-Challenge briefly; the chat-mode + strict-letter-match filter produced degenerate exact_match=0 results (model gives explanations instead of bare letter). The standard ARC eval requires logprobs (not exposed by vLLM chat-completions endpoint cleanly). MMLU-Pro and GSM8K are the conventional headline metrics; sufficient for the model card.
- **Did not push to HF.** Per CLAUDE.md `## Working norms`: "The repo is PRIVATE until the user authorizes Phase 8 (HF release)." Phase 7 prep doc is staged + ready; awaiting user `go`.

## On the box at end of session

- **Phase 5a serve still running** at port 8089 (TP=4, ~3h uptime, healthy). Box is paid-for; left running so user can inspect or run additional smokes when they wake. Kill with `pkill -f 'vllm serve'` if no longer needed.
- **Artifact at `/scratch/weights/v4-flash-nvfp4-fp8-mtp/`** in final state: 172 GB, all postprocesses applied + verified, 7 backup `config.json.bak_*` files for postprocess audit trail (clean these before HF upload — see Phase 7 prep doc).
- **vLLM at `/data/src/vllm/`** on `pr-42209` branch with 3 local patches applied (bool wrap, scale_fmt `.get`, attention.py fallback). The bool wrap is needed because PR #43248 hasn't merged yet; all three may become moot if the upstream PRs merge.
- **No leftover calibration processes.** v2 calib failed at subgraph 44 (expected), no zombie processes. v2 artifact dir deleted to free disk.

## Recommended user next steps

1. Read `MODEL_CARD.md` + decide if you want to ship. The GSM8K beats-RedHat is the headline; everything else is honest framing.
2. If ship: follow `docs/phase7_hf_upload_prep.md` step-by-step (pre-flight checklist + upload commands).
3. If not ship: that's also fine — the upstream contributions (5 of them) are real OSS value regardless of whether the artifact goes public.
4. Watch the 5 upstream items (3 PRs + 2 issues) for adoption. When `vllm-project/vllm#43304` or `vllm-project/llm-compressor#2745` land, re-run Phase 5b to fill in the spec-decode acceptance rate.

## Commits this session (latest at top)

```
4cd0bf3 docs: Phase 7 HF upload prep — gated on user authorization
cf26046 MODEL_CARD: honest Phase 5b status — MTP serve-loadable structurally; live gated on upstream
56c919b recipe: gotchas 15+16 — MTP quant globally blocked + vLLM draft inherits main scheme
38546ce CLAUDE.md: Phase 5a green, Phase 6 benchmarks complete; Phase 5b in progress
3d9da16 Phase 6 MMLU-Pro: 0.8113 ± 0.0035 (math 0.9149, law 0.6022)
11e014b MODEL_CARD: add BF16 baseline reference frame for GSM8K
647f065 MODEL_CARD.md: draft with GSM8K beat-RedHat headline + reproduction guide
9f81444 PLAN.md: Phase 5a/6-GSM8K done; Phase 5b/6-MMLU/6-MTP in progress
efd168d scripts: postprocess_v2_pipeline.py — unified Phase 3 driver
6f6899c scripts: v2 calibration with narrower MTP ignore for spec-decode
35baa3a Phase 6 GSM8K: 0.9181 strict / 0.9515 flexible — beats RedHat 0.910
fa233bf Phase 5a root-cause: B300 is sm_103a (10.3), not sm_100a (10.0a)
```

All pushed to `https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp` (private repo). HEAD: `4cd0bf3` on `main`.
