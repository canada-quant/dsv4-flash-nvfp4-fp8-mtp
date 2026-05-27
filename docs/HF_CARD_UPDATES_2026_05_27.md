# HF model card updates pending push (2026-05-27)

The 4 GitHub READMEs were updated this session. The 4 HuggingFace model cards still need parallel updates. Push these to HF when you have an HF_TOKEN available locally.

---

## Card B — `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`

**Add this section** to the HF README near the top (after the TL;DR / before quickstart):

```markdown
## NVFP4 execution reality on consumer Blackwell (SM 12.0) — honest naming ⚠️

The on-disk format is genuine NVFP4 — packed FP4 weights (uint8, 2 nibbles/byte) + FP8 E4M3 group scales + FP32 global scales. You get the **storage and GPU-memory-footprint win** of 4-bit weights wherever the artifact loads.

**On RTX PRO 6000 (SM 12.0) today, the actual MoE matmul does NOT use FP4 tensor cores.** With `VLLM_TEST_FORCE_FP8_MARLIN=1` (the required env var to load on consumer Blackwell), vLLM's NVFP4 oracle routes to `MarlinExperts` → `moe_wna16_marlin_gemm` with `b_q_type=scalar_types.float4_e2m1f`. Marlin dequantizes FP4→BF16 *inside the kernel* and the matmul runs on BF16 tensor cores. vLLM emits the warning: *"Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel."*

| | On B200/B300 (SM 10.0) | On RTX PRO 6000 (SM 12.0) today |
|---|---|---|
| Disk + GPU memory footprint | NVFP4 packed ✓ | NVFP4 packed ✓ (memory win preserved) |
| MoE expert matmul | Native FP4 tensor cores (tcgen05) | **Marlin BF16** (FP4→BF16 dequant inside kernel) |
| FLOPS / throughput vs BF16 | ~2× | **Same as BF16** — no FLOPS win |
| FP8 attention (separate path) | Native FP8 tensor cores | Native FP8 tensor cores ✓ |

For consumer-Blackwell users prioritizing **memory footprint**, Card B saves ~2× vs the W4A16 sibling. For **throughput**, Card B's expert path is functionally equivalent to Card D's W4A16 Marlin path on SM 12.0 — both bottom out in the same Marlin BF16 kernel.

Genuine NVFP4 tensor-core math on RTX PRO 6000 requires CUDA 13.0 + `compute_120f` cubins + FlashInfer SM120 fixes + vLLM's B12X auto-select to land. Open PRs tracking: [vllm-project/vllm#41738](https://github.com/vllm-project/vllm/pull/41738), [#43333](https://github.com/vllm-project/vllm/pull/43333), [#43341](https://github.com/vllm-project/vllm/pull/43341), [#43687](https://github.com/vllm-project/vllm/pull/43687). ETA: weeks-to-months.
```

**Also add** to the Limitations / known issues section:

```markdown
- **MTP × `--reasoning-parser deepseek_v4` × thinking-mode race** ([vllm-project/vllm#34650](https://github.com/vllm-project/vllm/issues/34650)): when serving with `--speculative-config method=mtp` + `--reasoning-parser deepseek_v4` + chat-template `thinking=True`, the structured-output `should_advance` race in `vllm/v1/structured_output/__init__.py:324-328` misses the `</think>` boundary and the response body comes back with `reasoning_content==''` AND `content==''` even though the model generated 200+ correct tokens. Workaround: drop `--reasoning-parser` for thinking-mode evals, or set `thinking=False`.
```

---

## Card D — `canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`

**Replace the AIME row** in the quality table (currently TBD or pre-fix numbers) with the verified 2026-05-27 result:

```markdown
| AIME-2024 c=4 thinking (RTX PRO 6000, jasl@27fd665b) | **24/30 = 80.0%** | 0 CUDA errors | 91.61% MTP acceptance, 641.9 s wall, finish_reasons {stop: 22, length: 8} |
```

Raw JSON: [`benchmarks/rtxpro6000/aime30_c4_thinking_jasl27fd665b.json`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp/blob/main/benchmarks/rtxpro6000/aime30_c4_thinking_jasl27fd665b.json) (committed today as part of the jasl#12 retest).

**Also add** to Limitations:

```markdown
- **MTP × reasoning_parser × thinking-mode race** ([vllm-project/vllm#34650](https://github.com/vllm-project/vllm/issues/34650)): same as the NVFP4 sibling. AIME-30 c=4 thinking workload bypasses this by using direct chat prompts without `chat_template_kwargs={"thinking":True}`; if you use that kwarg, the response body gets dropped to empty strings by the parser race. Workaround: drop `--reasoning-parser deepseek_v4` for thinking-mode runs.
```

---

## Card A — `canada-quant/DeepSeek-V4-Flash-W4A16-FP8`

**Add an early TL;DR-level banner**:

```markdown
> ⚠️ **Load status (2026-05-27):** Card A's published artifact is currently load-blocked on bleeding-edge vLLM (FP8 compressor scale naming + `e_score_correction_bias` architecture drift). Production deployments should pin `jasl/vllm@428e08e` (dual-Spark) or `jasl/vllm@ds4-sm120-experimental@abad5dc71` (2026-05-05) for the historical verified numbers below. For RTX PRO 6000 MTP, use the [W4A16-FP8-MTP sibling](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) — verified 2026-05-27 at 24/30 AIME-30 c=4 thinking, 0 CUDA errors, 91.6% MTP acceptance.
```

**Also scope-correct** the "1M context" claim explicitly to dual-DGX-Spark TP=2 (not H200 or RTX PRO 6000).

---

## Pro — `canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP`

**Add a Family / related repos section** (Pro is the only HF card missing reciprocal sibling links):

```markdown
## Family / related canada-quant DSv4 quants

| Card | HF | Role |
|---|---|---|
| **This** (V4-Pro NVFP4-FP8-MTP) | [Pro NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) | V4-Pro NVFP4 + MTP, **B200/B300-only**; +25-37% throughput vs upstream MXFP4 |
| Flash NVFP4-FP8-MTP | [NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) | smaller (V4-Flash 284B) — Blackwell-native + RTX PRO 6000 via Marlin-BF16 fallback |
| Flash W4A16-FP8-MTP | [W4A16-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) | Hopper-compatible MTP variant — H200 + RTX PRO 6000 (24/30 AIME c=4 thinking verified) |
| Flash W4A16-FP8 (no MTP) | [W4A16-FP8](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) | broadest hardware: H200 + DGX Spark + RTX PRO 6000 |
```

**Add a hardware-target scope note**: V4-Pro is B300-only by design (913 GiB artifact + NVFP4 oracle gates rule out consumer Blackwell).

---

## Quick decision matrix to add to ANY of the cards (pick whichever you think is most-read)

```markdown
## Which DSv4 canada-quant card on which hardware?

| Hardware | Recommended card | Notes |
|---|---|---|
| **B200 / B300 server** | **Pro** (V4-Pro) or **Flash NVFP4-FP8-MTP** | Real NVFP4 tcgen05 tensor-core math. Pro is 913 GiB; Flash is 172 GiB |
| **H200 (Hopper SM 9.0a)** | **Flash W4A16-FP8-MTP** (with MTP) or **Flash W4A16-FP8** (without) | Canonical Hopper path. NVFP4 not supported on Hopper |
| **RTX PRO 6000 Server Edition (SM 12.0), MTP + thinking-mode** | **Flash W4A16-FP8-MTP** | Verified 2026-05-27: 24/30 AIME-30 c=4 thinking, 0 errors, 91.6% MTP. Use jasl@27fd665b |
| **RTX PRO 6000, tight memory** | **Flash NVFP4-FP8-MTP** | NVFP4 storage win (~2× vs W4A16) but Marlin BF16 execution (no FLOPS win on SM 12.0) |
| **DGX Spark / GB10 (SM 12.1a)** | **Flash W4A16-FP8** | Dual-Spark TP=2 production canonical, 1M context |
| **Consumer single-RTX (5090 / consumer Blackwell)** | **Flash W4A16-FP8** or wait for B12X NVFP4 path | NVFP4 won't accelerate on consumer Blackwell yet |
```

---

## Push instructions

```bash
# Each HF card lives at a separate git LFS repo:
# 1. Clone each, with HF_TOKEN authenticated
git clone https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP hf-cardb
cd hf-cardb && (paste edits into README.md) && git add README.md
git commit -m "docs: NVFP4 storage-vs-execution honesty for SM 12.0 + #34650 caveat"
git push

# Repeat for the other 3 HF cards.
```

If you want me to also stage the exact full HF README files (vs deltas), let me know — I'd need to clone each HF card's README first, then produce the merged result.
