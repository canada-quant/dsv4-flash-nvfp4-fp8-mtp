# Findings index

One-line entry per finding doc, grouped by topic. Each links to a self-contained writeup.

## Methodology / measurement gotchas

- [MTP acceptance rate methodology (67% vs 85% disambiguation)](findings/mtp_acceptance_rate_methodology_2026_05_21.md) — same metric, different prompt format (raw `/v1/completions` vs `/v1/chat/completions`) yields ~20pt acceptance spread on the same content. Always disambiguate.
- [Thinking-mode plumbing verification](findings/thinking_mode_verification_2026_05_21.md) — DSV4 thinking is gated by `chat_template_kwargs.thinking + reasoning_effort`; 3 effective levels (off / high / max). Smoke confirms both quant + RedHat serves route correctly.

## vLLM / serve issues (and their fixes — see also docs/VLLM_SETUP_ISSUES.md)

- [vLLM `is_static_input_scheme` returns object, not bool](findings/vllm_bool_input_quant_dynamic.md) — defensive `bool()` wrap (PR #43248).
- [Phase 5a serve deferred / 4 layered mismatches](findings/phase5a_serve_deferred_2026_05_21.md) — config regex / `bool()` wrap / `weight_scale_inv` naming / sm_103a DeepGemm kernels. All resolved.

## Calibration friction

- [Sharded-MoE save coordination](findings/sharded_moe_save_coordination.md) — multi-rank artifact merge from per-rank subdirs to a unified safetensors layout.
- [Multi-rank Observer.synchronize hang](findings/multirank_observer_sync_hang.md) — async all_reduce desyncs on expert-sharded modules. Monkey-patch workaround.
- [4-rank A/B re-verify](findings/4rank_ab_re_verify_2026_05_20.md) — the gate test against the 1-rank baseline (keys / experts / MTP / weight-scale ratio).
- [Checkpointing design](findings/checkpointing_design_2026_05_20.md) — atomic .tmp+rename + resume-skip for the 10-12h calibration.
- [Phase 2b stall + completion](findings/phase2b_stall_2026_05_20.md), [Phase 2b complete](findings/phase2b_complete_2026_05_21.md) — 1-rank full calibration narrative.

## Upstream research / receipts

- [Upstream research 2026-05-20](findings/upstream_research_2026_05_20.md) — initial round of "what's already filed / merged / in PR" across vLLM, llm-compressor, compressed-tensors.
- [Upstream research 2026-05-21](findings/upstream_research_2026_05_21.md) — second round during Phase 5/6.

## Session summaries / context handoffs

- [Autonomous session summary 2026-05-21](findings/autonomous_session_summary_2026_05_21.md) — Phase 5/6 progress + handoff state.
- [Tier 1 static smoke 2026-05-20](findings/tier1_static_smoke_2026_05_20.md) — initial smoke before full calibration.

## Benchmark results — see docs/benchmarks/

| Benchmark | File |
|---|---|
| GSM8K (3-way, lm_eval) | [phase6_gsm8k_2026_05_21.md](benchmarks/phase6_gsm8k_2026_05_21.md) |
| MMLU-Pro (lm_eval) | [phase6_mmlu_pro_2026_05_21.md](benchmarks/phase6_mmlu_pro_2026_05_21.md) |
| HumanEval / IFEval (Phase 6 expanded) | [phase6_expanded_2026_05_21.md](benchmarks/phase6_expanded_2026_05_21.md) |
| Coding throughput (raw-completion) | [phase6_codebench_2026_05_21.md](benchmarks/phase6_codebench_2026_05_21.md) |
| Coding throughput (chat-template, 5 configs × 4 c × 2 TP) | [phase6_codebench_chat_sweep_2026_05_21.md](benchmarks/phase6_codebench_chat_sweep_2026_05_21.md) |
| **AIME 2024 thinking=high 3-way (the headline reasoning result)** | [tier1_aime24_2026_05_21.md](benchmarks/tier1_aime24_2026_05_21.md) |
