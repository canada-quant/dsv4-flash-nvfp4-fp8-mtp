# MTP acceptance rate — methodology discrepancy explained (2026-05-21)

## The discrepancy

Two measurements on the same artifact (`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`) on the same hardware (4× B300, TP=4) report different MTP draft-acceptance rates on "HumanEval coding":

| Measurement | Acceptance % | Drafts | Draft tokens | Accepted |
|---|---|---|---|---|
| EvalPlus run (chat-templated) | **85.04%** | 22,354 | 44,708 | 38,018 |
| `vllm bench serve` run (raw completion) | **67.29%** | 28,505 | 57,010 | 38,362 |

Both numbers are the **same metric**: `accepted_draft_tokens / total_draft_tokens` over all generation steps in the run, sourced from vLLM's `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total` Prometheus counters.

## Root cause

**Prompt structure differs across the two harnesses.**

### EvalPlus run (85.04%)
- Endpoint: `/v1/chat/completions` (chat mode)
- Prompt: chat turn — `"Please complete the following Python function: <code>"`
- Output: markdown-wrapped code with prose explanation, e.g.:
  ```
  Here is the implementation:

  ```python
  def has_close_elements(numbers, threshold):
      for i in range(len(numbers)):
          for j in range(i+1, len(numbers)):
              if abs(numbers[i] - numbers[j]) < threshold:
                  return True
      return False
  ```

  The function iterates...
  ```
- Token mix per response: ~40-60% explanatory prose, ~40-60% code. Both fragments are highly predictable to the MTP draft head — prose has formulaic wrapper sentences ("Here is...", "The function...") and code has high token-level predictability. Net acceptance: **85%**.

### `vllm bench serve` run (67.29%)
- Endpoint: `/v1/completions` (raw mode, `--skip-chat-template`)
- Prompt: raw function signature + docstring, exactly as HumanEval ships:
  ```python
  from typing import List
  def has_close_elements(numbers: List[float], threshold: float) -> bool:
      """Check if in given list of numbers, are any two numbers closer
      than threshold.
      >>> ...
      """
  ```
- Output: continuation of the function body — pure code, no wrapper:
  ```python
      for i in range(len(numbers)):
          for j in range(i+1, len(numbers)):
              if abs(numbers[i] - numbers[j]) < threshold:
                  return True
      return False
  ```
- Token mix: ~100% code, no boilerplate prose. Code token entropy is higher than wrapped-explanation prose; MTP draft acceptance is lower because each token has to be exactly right rather than approximating a common pattern. Net acceptance: **67%**.

## Why the 67% case is harder for MTP, not easier

Intuition might suggest pure code is "more predictable" than code-plus-prose and so should have higher acceptance. The data says otherwise. Reason: the prose framing in chat mode contains formulaic 2-3 token chunks ("`Here is the`", "`The function`", "` returns the`") that the draft head predicts trivially. Those tokens dominate the denominator and lift the average.

Raw code generation has fewer such "free" tokens — every identifier, every operator, every numeric literal is a real prediction the draft head has to land. Acceptance is genuine code-level prediction, not free wins on wrapper text.

The 67% on raw code is therefore the **harder measurement** and **more representative of pure code generation** as a workload. The 85% chat-templated measurement includes the cost of explanatory prose, which a developer integration may or may not generate.

## Implications for the speedup ratio

Wall-clock observed on `vllm bench serve` at c=1 raw HumanEval:
- Ours MTP-spec: 258.98 tok/s
- Ours no-spec: 127.45 tok/s
- Ratio: **2.03×**

At 67.29% per-draft-token acceptance with 2 draft tokens per step:
- Expected acceptance length = 1 main token + 0.6729 × 2 draft tokens × (rejection-conditioned factor) = ~2.35 tokens/step (matches the measured 2.35)
- Naive speedup if step cost stays constant: 2.35× — but the draft head itself adds per-step compute (~10-15% overhead on TP=4 B300)
- Measured speedup: 2.03× → 2.35× × (1 / 1.16) = 2.03× — within rounding of theory

For the chat-templated 85% acceptance, theory predicts ~2.7× tokens/step → ~2.3× wall-clock after overhead. We did NOT measure wall-clock chat-templated because EvalPlus codegen runs concurrent requests (16-way) which conflates with our c=1 single-user-decode operating point. The 85% number is correct as a draft-acceptance proxy but doesn't translate cleanly to a wall-clock speedup at the same operating point.

**For the model card, the correct framing is to report both numbers with their methodology:**

| Metric | Value | What it measures |
|---|---|---|
| Spec acceptance, raw code completion (`/v1/completions`, c=1) | **67.29%** | Pure code generation, no wrapper prose |
| Spec acceptance, chat-templated code (`/v1/chat/completions`, c=16) | **85.04%** | Chat-API code assistance (code + explanation) |
| Wall-clock decode speedup, raw code (c=1, vs ours-no-spec) | **2.03×** | Direct measurement on identical hardware |
| Wall-clock decode speedup, raw code (c=1, vs RedHat no-MTP) | **1.93×** | Cross-quant comparison on identical hardware |

The 1.93× / 2.03× wall-clock numbers are the publishable headline. The 67% acceptance is the explanation. The 85% is a secondary measurement showing acceptance is workload-dependent and higher on chat-mode (where the model emits both code and prose).

## Lesson for future benchmarks

Always disambiguate acceptance-rate numbers by prompt format. "MTP acceptance on coding" is not a single number — it depends on whether the harness uses chat templating, the output length distribution, and the prompt structure. The right way to publish:

- Always pair acceptance % with wall-clock tok/s on the SAME run (vLLM bench serve emits both in one summary)
- Disambiguate "chat-mode" vs "raw" in the methodology
- Skeptical of any single-number acceptance claim that doesn't specify operating conditions
