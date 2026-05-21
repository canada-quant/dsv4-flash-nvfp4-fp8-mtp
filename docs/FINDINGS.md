# Findings index

One-line entry per finding doc, grouped by topic. Each links to a self-contained writeup.

## Methodology / measurement gotchas

- [MTP acceptance rate methodology (67% vs 85% disambiguation)](findings/mtp_acceptance_rate_methodology_2026_05_21.md) — same metric, different prompt format (raw `/v1/completions` vs `/v1/chat/completions`) yields ~20pt acceptance spread on the same content. Always disambiguate.
- [Thinking-mode plumbing verification](findings/thinking_mode_verification_2026_05_21.md) — DSV4 thinking is gated by `chat_template_kwargs.thinking + reasoning_effort`; 3 effective levels (off / high / max).

## vLLM / serve issues (and their fixes — see also docs/VLLM_SETUP_ISSUES.md)

- [vLLM `is_static_input_scheme` returns object, not bool](findings/vllm_bool_input_quant_dynamic.md) — defensive `bool()` wrap (PR #43248).

## Calibration friction

- [Sharded-MoE save coordination](findings/sharded_moe_save_coordination.md) — multi-rank artifact merge from per-rank subdirs to a unified safetensors layout.
- [Multi-rank Observer.synchronize hang](findings/multirank_observer_sync_hang.md) — async all_reduce desyncs on expert-sharded modules. Monkey-patch workaround.
- [Checkpointing design](findings/checkpointing_design_2026_05_20.md) — atomic .tmp+rename + resume-skip for the long calibration run.

## Benchmark results — see docs/benchmarks/

| Benchmark | File |
|---|---|
| GSM8K (3-way, lm_eval) | [phase6_gsm8k_2026_05_21.md](benchmarks/phase6_gsm8k_2026_05_21.md) |
| MMLU-Pro (lm_eval) | [phase6_mmlu_pro_2026_05_21.md](benchmarks/phase6_mmlu_pro_2026_05_21.md) |
| HumanEval / IFEval | [phase6_expanded_2026_05_21.md](benchmarks/phase6_expanded_2026_05_21.md) |
| Coding throughput (raw-completion) | [phase6_codebench_2026_05_21.md](benchmarks/phase6_codebench_2026_05_21.md) |
| Coding throughput (chat-template, 5 configs × 4 c × 2 TP) | [phase6_codebench_chat_sweep_2026_05_21.md](benchmarks/phase6_codebench_chat_sweep_2026_05_21.md) |
| AIME 2024 thinking=high 3-way | [tier1_aime24_2026_05_21.md](benchmarks/tier1_aime24_2026_05_21.md) |
