# DeepSeek-V4-Flash thinking-mode plumbing — verification (2026-05-21)

Front-loaded gate before running hard reasoning benchmarks (AIME / GPQA / LiveCodeBench). Confirms what the thinking-mode parameter is, what levels exist, and that both `ours-MTP-spec` and `RedHat` serves route it correctly.

## How thinking-mode is activated in vLLM

vLLM's DSV4 chat template (in `/data/src/vllm/vllm/tokenizers/deepseek_v4.py` + `deepseek_v4_encoding.py`) reads two `chat_template_kwargs` keys:

- `thinking: bool` (or `enable_thinking: bool`, same effect)
- `reasoning_effort: str` — `"none"`, `"high"`, `"max"` (or `"xhigh"`, alias for `"max"`)

The combined logic:

| `thinking` | `reasoning_effort` | Effective mode |
|---|---|---|
| false / unset | any | **chat** (no thinking) |
| true | unset / `"none"` | chat (thinking is overridden off by `"none"`) |
| true | unset (default) → resolves to `"high"` if `thinking=true` | **thinking @ high** |
| true | `"high"` | **thinking @ high** |
| true | `"max"` / `"xhigh"` | **thinking @ max** — prepends `REASONING_EFFORT_MAX` system prompt |

So three effective serving modes: **off**, **high**, **max**.

The `REASONING_EFFORT_MAX` prefix (`/data/src/vllm/vllm/tokenizers/deepseek_v4_encoding.py:70`):

> "Reasoning Effort: Absolute maximum with no shortcuts permitted.
> You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
> Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked."

This system prompt adds ~78 tokens to the input. The user-facing parameter goes via the OpenAI-compatible chat endpoint as:

```json
{"model": "...", "messages": [...],
 "chat_template_kwargs": {"thinking": true, "reasoning_effort": "max"}}
```

## Smoke test results — same prompt across both serves × 3 levels

**Prompt**: `"Solve: if x^2 - 5x + 6 = 0, find both values of x. Show your reasoning."` (31 input tokens)

| Serve | Mode | Prompt tokens | Completion tokens | Final answer correct |
|---|---|---|---|---|
| `ours-MTP-spec` (TP=4, NVFP4+MTP) | off | 31 | 250 | yes (x=2 or 3) |
| `ours-MTP-spec` | high | 31 | 293 | yes |
| `ours-MTP-spec` | max | **110** (+79) | 324 | yes |
| `RedHat` (TP=4, NVFP4 no-MTP) | off | 31 | 185 | yes |
| `RedHat` | high | 31 | 320 | yes |
| `RedHat` | max | **110** (+79) | 486 | yes |

Both serves:
- Accept `chat_template_kwargs` without error
- Show the expected prompt-token jump (+78-79) for `reasoning_effort=max` (confirms the system prompt is prepended)
- Generate progressively longer outputs as effort increases
- Produce the correct final answer (`x = 2 or 3`) at every level

**No `<think>` tags appeared in output** because the serve was started without `--reasoning-parser deepseek_v4`. With that flag, vLLM's `DeepSeekV4ReasoningParser` would split reasoning trace into a separate `message.reasoning` field. Without it, all reasoning is in `message.content`. This is purely presentational — benchmark harnesses (lm_eval, EvalPlus) extract the final answer via regex from the full output regardless.

## Implications for the existing Phase 6 benchmarks

The earlier benchmarks (GSM8K, MMLU-Pro, HumanEval, IFEval, throughput sweeps) were **all run with default chat-template kwargs** — i.e., **thinking OFF**. None of those harnesses pass `chat_template_kwargs.thinking=true`:

- `lm_eval gsm8k` / `lm_eval mmlu_pro` with `--apply_chat_template`: applies chat template but doesn't inject thinking kwargs.
- `lm_eval ifeval` with `--apply_chat_template`: same.
- `EvalPlus codegen --backend openai`: hits `/v1/chat/completions` with no `chat_template_kwargs`.
- `vllm bench serve --backend openai-chat`: no `chat_template_kwargs` passed.

**The Phase 6 published numbers reflect non-thinking inference.** This needs to be stated in the model card. Adding thinking-mode equivalent numbers for these benchmarks would be a separate measurement campaign — feasible but ~2-3× more bench time per benchmark due to longer outputs.

## Why this matters for the speedup claim

The MTP acceptance rate is **content-dependent**:
- 67% on raw code completion (high-density code tokens)
- 88% on chat-templated code (code + wrapper prose)
- Unknown on thinking-mode reasoning traces (longer, more diverse, math/logic-heavy)

If MTP acceptance on thinking-mode AIME drops to 70%, the wall-clock speedup claim narrows for reasoning workloads. If it holds at 85%+, the artifact positions as fast on both coding AND reasoning — a stronger claim. This is the specific unknown that motivates running Tier 1 with acceptance capture.

## Recommended Tier 1 thinking config

For AIME / GPQA / LiveCodeBench:
- **thinking=true, reasoning_effort=high** (the default "thinking on" mode)
- Capture MTP acceptance per benchmark
- Skip `reasoning_effort=max` for first pass (longer outputs → more bench time, lower acceptance expected)
- Run on `ours-MTP-spec @ TP=4` vs `RedHat @ TP=4` (TP=4 confirmed as the better operating point for our artifact in Phase 6 chat-sweep)

## Verdict

**Gates (a), (b), (c) PASSED:**
- (a) Thinking parameter identified: `chat_template_kwargs.thinking` + `chat_template_kwargs.reasoning_effort` — 3 levels (off / high / max). Documented.
- (b) Smoke test confirms both serves route correctly: prompt-token deltas match expected system-prompt injection at `max`; output length scales with effort; final answer correctness preserved at every level on both serves.
- (c) Existing Phase 6 benchmarks audited: all ran with thinking OFF (default chat_template_kwargs). To be noted in model card.

**Cleared to proceed with Tier 1** at `thinking=true, reasoning_effort=high`, capturing MTP acceptance per benchmark.
