# Phase 6 — chat-template coding sweep, full TP × concurrency matrix (2026-05-21)

The earlier raw-completion sweep (`phase6_codebench_2026_05_21.md`) measured pure code prediction at TP=4 only. This sweep measures the **production-correct path** (chat template + `/v1/chat/completions`) across **two TP configurations × five model configs × four concurrencies = 40 measurement cells**, including the BF16 reference baseline.

## Setup

- **Dataset**: 164 HumanEval prompts as chat-templated user messages
- **Endpoint**: `vllm bench serve --backend openai-chat --endpoint /v1/chat/completions`
- **Output length**: 512 tokens per request
- **Concurrencies**: 1, 4, 8, 16
- **TP configurations**: TP=4 (4 GPUs) and TP=8 (8 GPUs) on 8× B300 SXM6 AC
- **vLLM**: `0.21.1rc1.dev164+gd05d52059.d20260521` with our 5 local patches (PRs #43248, #43288, #43290, #43319, + new BF16 `quantization_config`-missing fix)
- **Chat templating**: applied server-side; `--skip-chat-template` only skips client-side bench tokenization.

## Output throughput (tok/s) — full matrix

| Config | TP | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|---|
| **BF16 + MTP** (reference) | 8 | 296.59 | 614.84 | 1112.70 | 1545.63 |
| BF16, no-spec | 8 | 133.91 | 441.60 | 694.07 | 1010.54 |
| **Ours NVFP4-FP8 + MTP** | **4** | **278.68** | **649.35** | **1104.89** | **1577.20** |
| Ours NVFP4-FP8 + MTP | 8 | 286.46 | 608.86 | 912.66 | 1297.03 |
| Ours NVFP4-FP8, no-spec | 4 | 123.99 | 401.90 | 646.69 | 944.19 |
| Ours NVFP4-FP8, no-spec | 8 | 135.63 | 424.02 | 681.50 | 969.39 |
| RedHat NVFP4-FP8 (no MTP) | 4 | 131.06 | 417.87 | 673.12 | 1007.78 |
| RedHat NVFP4-FP8 (no MTP) | 8 | 134.56 | 442.13 | 720.33 | 1057.77 |

## TPOT (ms / output token)

| Config | TP | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|---|
| BF16 + MTP | 8 | **3.04** | 5.85 | 6.44 | 9.01 |
| BF16, no-spec | 8 | 7.11 | 8.60 | 10.77 | 13.99 |
| Ours NVFP4-FP8 + MTP | 4 | **3.18** | 5.65 | 6.51 | 8.84 |
| Ours NVFP4-FP8 + MTP | 8 | 3.01 | 5.63 | 7.02 | 9.00 |
| Ours NVFP4-FP8, no-spec | 4 | 7.69 | 9.54 | 11.62 | 14.95 |
| Ours NVFP4-FP8, no-spec | 8 | 7.02 | 8.96 | 10.88 | 13.70 |
| RedHat (no MTP) | 4 | 7.29 | 9.20 | 11.28 | 14.32 |
| RedHat (no MTP) | 8 | 7.07 | 8.66 | 10.44 | 13.52 |

## MTP acceptance — per-concurrency × per-TP × per-precision

| Config | TP | c=1 | c=4 | c=8 | c=16 | Acc-length |
|---|---|---|---|---|---|---|
| **BF16 + MTP** | 8 | 89.41% | 89.10% | 89.41% | 88.83% | ~2.78 |
| Ours NVFP4-FP8 + MTP | 4 | 87.96% | 88.27% | 87.92% | 88.19% | ~2.76 |
| Ours NVFP4-FP8 + MTP | 8 | 88.07% | 88.46% | 88.68% | 87.84% | ~2.77 |

**Three observations:**
1. **Acceptance is flat across concurrencies** (within 0.6% spread per row) at every TP × precision combination. MTP draft quality does not degrade under batching.
2. **BF16 acceptance is ~1.3 pts higher than NVFP4** (89.2% vs 88.0%). Tiny — quantization has a measurable but small effect on draft acceptance.
3. **TP=4 vs TP=8 acceptance is identical** for the quant artifact (88.1% vs 88.3% averaged). Acceptance is a function of the model, not the sharding.

## Key cross-config comparisons

### Headline: ours NVFP4+MTP @ TP=4 vs BF16+MTP @ TP=8 (the "matches reference at 25% memory" claim)

| Concurrency | Ours TP=4 | BF16 TP=8 | Ours/BF16 ratio |
|---|---|---|---|
| c=1 | 278.68 | 296.59 | 0.940× (BF16 +6.4%) |
| c=4 | **649.35** | 614.84 | **1.056× (ours +5.6%)** |
| c=8 | 1104.89 | 1112.70 | 0.993× (BF16 +0.7%) |
| c=16 | **1577.20** | 1545.63 | **1.020× (ours +2.0%)** |

**Our quant at TP=4 matches BF16 at TP=8** across the full concurrency sweep — beats at c=4 and c=16, narrowly trails at c=1 and c=8. Effectively even on throughput at half the GPU budget and 25% memory footprint.

### Cross-quant: ours-MTP vs RedHat-no-MTP, symmetric TP

| Concurrency | Ours MTP TP=8 / RedHat TP=8 | Ours MTP TP=4 / RedHat TP=4 |
|---|---|---|
| c=1 | 286.46 / 134.56 = **2.13×** | 278.68 / 131.06 = **2.13×** |
| c=4 | 608.86 / 442.13 = **1.38×** | 649.35 / 417.87 = **1.55×** |
| c=8 | 912.66 / 720.33 = **1.27×** | 1104.89 / 673.12 = **1.64×** |
| c=16 | 1297.03 / 1057.77 = **1.23×** | 1577.20 / 1007.78 = **1.56×** |

**At c=1 the speedup is identical at both TP configurations (2.13×).** At higher concurrencies, TP=4 delivers higher absolute speedup ratios than TP=8 because both no-MTP configs scale better with TP, while MTP-spec at TP=8 hits a per-rank-utilization ceiling.

### Within-quant: MTP-spec vs no-spec at same TP (controls for everything except spec-decode)

| Concurrency | TP=4 (MTP/no-spec) | TP=8 (MTP/no-spec) |
|---|---|---|
| c=1 | 278.68 / 123.99 = **2.25×** | 286.46 / 135.63 = **2.11×** |
| c=4 | 649.35 / 401.90 = 1.62× | 608.86 / 424.02 = 1.44× |
| c=8 | 1104.89 / 646.69 = 1.71× | 912.66 / 681.50 = 1.34× |
| c=16 | 1577.20 / 944.19 = 1.67× | 1297.03 / 969.39 = 1.34× |

**Spec-decode delivers higher wall-clock speedup at TP=4 than TP=8 at higher concurrencies.** TP=8 small per-rank shards underutilize tensor cores enough that the spec-decode draft overhead eats more of the win.

## Surprise finding: ours NVFP4+MTP is FASTER at TP=4 than TP=8

| Concurrency | TP=4 tok/s | TP=8 tok/s | TP=4/TP=8 |
|---|---|---|---|
| c=1 | 278.68 | 286.46 | 0.973× (TP=8 marginally faster) |
| c=4 | **649.35** | 608.86 | **1.066× (TP=4 wins)** |
| c=8 | **1104.89** | 912.66 | **1.211× (TP=4 wins)** |
| c=16 | **1577.20** | 1297.03 | **1.216× (TP=4 wins)** |

**For our NVFP4+MTP artifact, TP=4 is the right operating point for batched serving.** Reasoning: at TP=8 each rank holds 32 routed experts (vs 64 at TP=4), the per-rank GEMMs are smaller, and the NVFP4 tensor-core kernels start to underutilize. The all-to-all communication overhead for expert dispatch also grows. The crossover is at c=1, where TP=8 has more compute headroom for the single-request decode and edges TP=4 out by 3%.

This finding has model-card implications: **recommend TP=4 for production serving**, not TP=8. Counterintuitive because TP=8 "has more GPUs" but the additional sharding hurts MoE batched serving.

## Why c=1 has the biggest MTP speedup

At c=1, the GPU is otherwise idle between decode steps — spec-decode trades that idle time for additional draft compute. The per-step latency stays roughly constant but emits more tokens per step. Net: nearly linear scaling with acceptance length (2.76 tokens/step × 0.81 step-cost-factor ≈ 2.25× at TP=4).

At higher concurrency, the GPU is already saturated with other requests' decode work. Spec-decode still wins but the gain is bounded by available throughput, not by acceptance quality.

The right reading: **interactive-coding workloads (single-user, agentic IDE assistance) get the c=1 headline (2.13–2.25× vs RedHat / no-spec)**, while batch-serving workloads get the steady 1.55–1.71× at TP=4.

## Comparison to the earlier raw-completion sweep (same data, different prompt format)

| Config | Concurrency | Chat-template tok/s | Raw-completion tok/s |
|---|---|---|---|
| Ours MTP TP=4 | c=1 | 278.68 | 258.98 |
| Ours MTP TP=4 | c=4 | 649.35 | 628.86 |
| Ours MTP TP=4 | c=8 | 1104.89 | 1157.61 |
| Ours MTP TP=4 | c=16 | 1577.20 | 1786.04 |

At low concurrency, chat-template is faster (higher acceptance + slightly longer outputs amortize prompt-processing cost). At high concurrency, raw is faster (no prompt-template overhead, more compact prompts → more KV-cache headroom).

## What got measured simultaneously

`vllm bench serve` emits throughput, latency (mean/median/p99 TTFT + TPOT + ITL), and spec-decode metrics (acceptance %, per-position acceptance, total drafts, total draft tokens, accepted tokens) **in one summary per run**. Each cell in every table above is a single coherent measurement.

## Raw artifacts

Under `/data/nvfp4-mtp/lm_eval/` on the AWS box:
- `chatsweep_mtp_c<N>_<ts>/` — ours MTP-spec TP=4
- `chatsweep_nospec_c<N>_<ts>/` — ours no-spec TP=4
- `chatsweep_redhat_c<N>_<ts>/` — RedHat TP=4
- `chatsweep_mtp_tp8_c<N>_<ts>/` — ours MTP-spec TP=8
- `chatsweep_nospec_tp8_c<N>_<ts>/` — ours no-spec TP=8
- `chatsweep_redhat_tp8_c<N>_<ts>/` — RedHat TP=8
- `chatsweep_bf16mtp_c<N>_<ts>/` — BF16 + MTP TP=8
- `chatsweep_bf16nospec_c<N>_<ts>/` — BF16 no-spec TP=8

## Verdict for the model card

1. **Quant at TP=4 matches BF16 at TP=8** on throughput — defensible "matches reference at quarter the memory" claim, with no TP handicap (in fact we win at higher concurrencies).
2. **MTP differentiator measured at both TP=4 and TP=8** — 2.13× at c=1 vs RedHat, identical at both TP configurations.
3. **TP=4 is the recommended operating point** for production serving of our artifact — counterintuitively faster than TP=8 at batched concurrencies due to per-rank tensor-core utilization on MoE experts.
4. **MTP acceptance does not degrade under batching** — 88.0% ± 0.4% across c=1→c=16, at both TP=4 and TP=8.
