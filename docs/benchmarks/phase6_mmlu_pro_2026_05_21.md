# Phase 6 — MMLU-Pro benchmark results (2026-05-21)

## Headline

**`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`** scores **0.8113 ± 0.0035** on MMLU-Pro (5-shot, custom-extract, full 12032 test set).

| Run | MMLU-Pro (5-shot, EM) |
|---|---|
| **This artifact (NVFP4-FP8-MTP)** | **0.8113 ± 0.0035** |
| DeepSeek-V4-Flash BF16 (paper-reported, 5-shot) | 0.875–0.910 (varies by report column) |
| `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` | not reported |

## Subject breakdown

| Subject | EM ± stderr |
|---|---|
| biology | 0.8996 ± 0.0112 |
| business | 0.8758 ± 0.0117 |
| chemistry | 0.8525 ± 0.0105 |
| computer_science | 0.8488 ± 0.0177 |
| economics | 0.8590 ± 0.0120 |
| engineering | 0.7110 ± 0.0146 |
| health | 0.7897 ± 0.0143 |
| history | 0.7034 ± 0.0234 |
| **law** | **0.6022 ± 0.0148** ← lowest |
| **math** | **0.9149 ± 0.0076** ← highest |
| other | 0.7803 ± 0.0136 |
| philosophy | 0.7455 ± 0.0195 |
| physics | 0.8599 ± 0.0096 |
| psychology | 0.8459 ± 0.0128 |
| **overall** | **0.8113 ± 0.0035** |

The gap between the lowest (law 0.60) and highest (math 0.91) is consistent across DSV4-Flash reports — DSV4 is known to be strong on math/STEM and weaker on law/legal-language. The quantization doesn't appear to skew this distribution.

## Run details

- Date: 2026-05-21, 08:25:02 UTC start, 09:05:14 UTC end (~40 min wall)
- Hardware: AWS p6-b300.48xlarge, B300 SXM6 AC (sm_103a), 4× GPU TP
- vLLM: `0.21.1rc1.dev164+gd05d52059.d20260521` (mainline + PR #42209 + 3 local patches)
- Build: `TORCH_CUDA_ARCH_LIST=10.3a`
- Serve config: `--tensor-parallel-size 4 --kv-cache-dtype fp8` (no `--speculative-config`)
- Eval harness: `lm-eval 0.4.12` via `local-chat-completions`
- Task: `mmlu_pro`, num_fewshot=5 (lm_eval default for MMLU-Pro), batch_size=1, num_concurrent=16, apply_chat_template=True
- Sample count: 12,032 (full test set across all subjects)

## Caveats

- DeepSeek's reported MMLU-Pro numbers vary across their docs (73.5, 87.5, 89.1, 91.0) depending on the source table and the specific config (think-mode, format, etc.). Without a controlled like-for-like BF16 baseline on this exact eval config, the gap between our 0.8113 and DeepSeek's high-end 0.91 can't be attributed cleanly to quantization vs eval-procedure differences.
- RedHat doesn't publish MMLU-Pro for `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`, so no direct quant-vs-quant comparison is available.
- Our run uses `apply_chat_template=True`. DeepSeek's official evals may use raw-prompt format. Format mismatch can move MMLU-Pro by several points.

Raw results JSON: `docs/benchmarks/mmlu_pro_2026_05_21.json`.

## Reproduction

```bash
# With serve up at port 8089:
lm_eval run \
  --model local-chat-completions \
  --tasks mmlu_pro \
  --model_args "model=$ARTIFACT_DIR,base_url=http://localhost:8089/v1/chat/completions,num_concurrent=16,max_retries=3,tokenized_requests=False" \
  --batch_size 1 \
  --apply_chat_template \
  --output_path mmlu_pro_results
```

## Status

Phase 6 (no spec-decode) complete:
- GSM8K: 0.9181 strict / 0.9515 flexible — beats RedHat 0.910
- MMLU-Pro: 0.8113 overall, math 0.9149 / law 0.6022 (typical DSV4 spread)

Pending:
- Phase 5b spec-decode acceptance rate (gated on v2 calibration with FP8 MTP attn completing — in progress on GPU 4)
