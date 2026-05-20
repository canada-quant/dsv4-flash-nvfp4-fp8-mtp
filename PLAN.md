# PLAN.md — DeepSeek-V4-Flash NVFP4-FP8-MTP

## Goal

Two tracks, both shipping under the `canada-quant` org on GitHub *and* HF (`huggingface.co/canada-quant`).

**Track A — the artifact.** Ship **the first** DeepSeek-V4-Flash quantization that combines:
- NVFP4 routed MoE experts (Blackwell-native, matches RedHat's recipe)
- FP8_BLOCK 128×128 attention (matches RedHat's recipe)
- **Preserved MTP layer** (RedHat dropped this; we don't)

Target HF artifact: `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`

**Track B — upstream contributions.** Every non-trivial bug we hit while shipping the artifact is also a contribution candidate. The strategic point: a quant lab that *only* ships re-quantizations is one of dozens; a lab that ships re-quantizations *and* fixes the bugs that surfaced during the work is taken seriously. Currently queued (filed under `canada-quant` org / paul@sunday.market identity):

| # | Repo | Bug | Status |
|---|---|---|---|
| 1 | `vllm-project/vllm` | W4A16 MoE TP sharding (already filed as issue #41511) | issue open, PR not yet |
| 2 | `vllm-project/vllm` | `bool()` on `input_quant and not input_quant.dynamic` returns `None` when `input_quant=None` (compressed_tensors.py:683,704) | not yet filed |
| 3 | `neuralmagic/llm-compressor` | `Observer.synchronize` desyncs across ranks when modules are expert-sharded (`base.py:157` + `moving_base.py:123`) — surfaced by *this* artifact's 4-rank dryrun | not yet filed; monkey-patch in this repo until merged |
| 4 | (TBD) | FP8 Marlin SM 10.0 capability detection — depends on whether it's a genuine bug | needs reproduction first |

The monkey-patch we land for #3 in this repo is the same code change as the upstream PR, just packaged differently. We ship the artifact on the monkey-patch path, then file the PR with the artifact as the real-world reproducer in the PR description.

## Phase status

| # | Phase | Status | Owner | ETA |
|---|---|---|---|---|
| 0 | Bootstrap p6-b300 box | reusing sibling repo's venvs | done | n/a |
| 1 | BF16 dequant (`/scratch/weights/bf16-mtp`) | already exists from sibling repo | done | n/a |
| 2a | NVFP4 multi-rank dryrun (verify no NCCL hang) | **done** — gate passed 2026-05-20 with observer-sync monkey-patch + multi-rank save fix | calib | 30 min |
| 2b | NVFP4 full 8-rank calibration | pending — gated on per-layer checkpointing + 4-number A/B re-verify | calib | 10-12h |
| 3 | Post-process artifact for vLLM | scaffolded (transfers from sibling) | post | 30 min |
| 4 | Verify (MTP keys, MTP weights quantized) | scripts ready | verify | 15 min |
| 5 | vLLM serve smoke + harness battery | **harness chosen**: jasl/vllm-ds4-sm120-harness (see Harness section below) | serve | 1h smoke + 2h harness |
| 6 | Benchmark vs RedHat (GSM8K, MMLU-Pro) + MTP acceptance rate (the headline diff metric vs RedHat) | TBD | bench | 4-8h |
| 7 | Model card + README + HF upload | TBD | release | 2h |
| 8 | Public release (gated on user authorization) | pending | release | n/a |

## Harness choice (settled 2026-05-20)

Phase 5 (serve smoke + correctness gate) uses **`jasl/vllm-ds4-sm120-harness`** cloned at `/data/harness/` on the B300 box. The `sm120` in the name is misleading — the harness is **hardware-agnostic on the client side**, runs against any OpenAI-compatible vLLM server, and has first-class support for the exact thing this artifact differentiates on:

- `--speculative_config '{"method":"mtp","num_speculative_tokens":2}'` is a first-class harness variant.
- Checked-in B200 TP=4 baseline bundle at `baselines/20260502_b200_tp4_main_5737770c6/` with both `nomtp` and `mtp` reference data for behavior-compare.
- `generation-compare` diffs our NVFP4-MTP output against the BF16-MTP baseline; quantization-induced token-level divergence is expected, behavior-shape match is the gate.
- `lm-eval` integration for GSM8K (the metric RedHat reports as 0.910).
- **Runtime telemetry captures MTP acceptance rate from the vLLM server log** — this is the headline proof-of-life number for the artifact. RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` cannot report this number (no MTP weights), so MTP acceptance rate IS the differentiator metric in the model card.

Caveats (do not re-debate next session):
- B200 baseline is BF16, NVFP4 artifact will diverge at token level. Use `generation-compare` + `lm-eval` for quality, not `oracle-compare` for strict token oracle.
- B300 serve flags: default `--attention_config.use_fp4_indexer_cache=True` (B300 is same FP4-tensor-core family as B200/SM100). Flip OFF only if vLLM rejects it. Do not blindly copy SM12x profiles from harness configs.
- MMLU-Pro is `lm_eval --task mmlu_pro` — not a checked-in harness task but the harness composes lm-eval, so add it as a `scripts/run_lm_eval.sh` override.
- Sibling W4A16 repo's `smoke_test_adapter.py` / `smoke_gptq_real.py` (at `/data/scripts/`) are NOT the right gate for this artifact — wrong recipe family. Do not use.

## Smoke ordering (4 tiers, do not skip)

1. **Static smoke (no GPU)** — inspect dryrun artifact `config.json` (`num_nextn_predict_layers: 1`, `quantization_config` shape), `recipe.yaml`, confirm 256 expert paths + 799 MTP keys + scale dtypes (FP8 e4m3fn for FP8_BLOCK attn, FP4 packed for NVFP4 experts). `vllm.model_executor.layers.quantization.compressed_tensors.CompressedTensorsConfig.from_config()` parse test. Catches ~80% of config-format breaks. Runs concurrent with calibration.
2. **Harness prep (no GPU)** — `.env` configured, `pytest -q tests` dry-run against a placeholder server confirms harness itself works. Pull B200 TP=4 baseline bundle. Concurrent with calibration.
3. **vLLM load smoke (post-calibration, blocks on GPU availability)** — TWO load attempts. First WITHOUT `--speculative_config` (does the artifact serve at all?). Second WITH `--speculative_config method=mtp` (does the MTP draft head load AND produce a measurable acceptance rate?). The second attempt IS the differentiator demonstration.
4. **Full harness battery (Phase 5)** — chat-smoke, generation-matrix `--variant mtp`, toolcall15, `lm-eval --task gsm8k`, `lm-eval --task mmlu_pro`, `oracle-compare` for behavior shape, plus MTP acceptance-rate extraction from `runtime_stats_summary.json` into the model card.

## Recipe (locked)

```python
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import NVFP4, FP8_BLOCK
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[
                r"re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
                r"re:.*mtp\.\d+\.(e_proj|h_proj)$",
            ],
            format="float-quantized",
            **FP8_BLOCK,
        ),
        "experts": QuantizationScheme(
            targets=[r"re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$"],
            format="nvfp4-pack-quantized",
            **NVFP4,
        ),
    },
    ignore=[
        "head", "embed",
        r"re:.*norm.*",
        r"re:.*\.ffn\.gate$",
        r"re:.*\.ffn\.gate\..*",
        r"re:.*\.ffn\.shared_experts\..*",
        r"re:.*\.hc_.*",
        r"re:hc_.*",
        r"re:.*\.attn\.attn_sink$",
        r"re:.*\.attn\.(compressor|indexer)\..*",
    ],
)
```

**Calibration corpus (locked, matches predecessor exactly):**
- `HuggingFaceH4/ultrachat_200k`
- split `train_sft`, seed 42, **768 samples**, max_seq_len **512**

## Topology (locked pending dryrun verification)

- 4-rank dryrun first (8 samples, 1 layer in recipe, ~5-10 min wall)
- If 4-rank dryrun completes without hanging at any collective: 8-rank full run
- If 4-rank dryrun hangs: fall back to 1-rank full run (~8-12h on a single B300)

## Out of scope for this repo

- W4A16 GPTQ recipe → handled by sibling `canada-quant/dsv4-flash-w4a16-fp8-mtp`
- V4-Pro → separate repo when V4-Flash NVFP4-MTP ships (likely `canada-quant/dsv4-pro-nvfp4-fp8-mtp`)
- vLLM-side kernel patches → already validated in sibling repo's H200 work, not touched here

## Risk register

| risk | impact | likelihood | mitigation |
|---|---|---|---|
| `QuantizationModifier` ALSO has cross-rank collectives we haven't audited (observer stats sync, compressed_tensors broadcast) | full run hangs same as GPTQ | medium | 4-rank dryrun before full run; fall back to 1-rank if hangs |
| NVFP4 expert weight shape doesn't match vLLM's `compressed_tensors_moe_w4a4_nvfp4` expected layout | serve fails to load | medium | savetest-style smoke load before claiming complete |
| MTP layer's MoE experts get quantized correctly but the entry projections (`e_proj`, `h_proj`) need different scheme | spec-decode draft produces garbage | low | verify via cosine similarity smoke after calibration |
| Spot reclamation mid-run | partial loss | medium | calibration is short (4-12h) vs 3-day reservation; no checkpointing but low exposure window |
| RedHat ships V4-Flash NVFP4 with MTP after us | differentiator weakened | low | they'd need to fork transformers; no indication they will |

## What transfers from sibling repo

- `patches/modeling_deepseek_v4.py.diff` (MTP key retention) — load-bearing
- `patches/helpers.py.diff` (dotless tensor match fix) — load-bearing
- `vendor/dsv4-upstream/` (verbatim upstream model.py + kernel.py)
- `scripts/upstream/__init__.py` (GPTQLinear, decoupled expert sharding, dist-mask around Transformer construction)
- `scripts/load_bf16_into_transformer.py`
- `scripts/calibration_model.py`
- `scripts/postprocess_for_vllm.py` (config field merge, MTP key rename, scale_fmt — may need NVFP4-specific tweaks)
- `scripts/verify_mtp_keys.py`, `scripts/verify_mtp_quantized.py`

## What's new in this repo

- `scripts/quantize_v4_nvfp4_fp8_mtp.py` — recipe + entry point using `QuantizationModifier` instead of `GPTQModifier`
- Different post-save expectations (vLLM NVFP4 backend instead of WNA16)
- Different model card framing (MTP vs RedHat, not GPTQ-vs-RTN-vs-predecessor)
