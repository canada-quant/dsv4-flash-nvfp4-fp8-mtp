# PLAN.md — DeepSeek-V4-Flash NVFP4-FP8-MTP

## Goal

Ship **the first** DeepSeek-V4-Flash quantization that combines:
- NVFP4 routed MoE experts (Blackwell-native, matches RedHat's recipe)
- FP8_BLOCK 128×128 attention (matches RedHat's recipe)
- **Preserved MTP layer** (RedHat dropped this; we don't)

Target HF artifact: `canada-quant-labs/DeepSeek-V4-Flash-NVFP4-FP8-MTP`

## Phase status

| # | Phase | Status | Owner | ETA |
|---|---|---|---|---|
| 0 | Bootstrap p6-b300 box | reusing sibling repo's venvs | done | n/a |
| 1 | BF16 dequant (`/scratch/weights/bf16-mtp`) | already exists from sibling repo | done | n/a |
| 2a | NVFP4 multi-rank dryrun (verify no NCCL hang) | **next** | calib | 30 min |
| 2b | NVFP4 full 8-rank calibration | gated on 2a | calib | 4-12h |
| 3 | Post-process artifact for vLLM | scaffolded (transfers from sibling) | post | 30 min |
| 4 | Verify (MTP keys, MTP weights quantized) | scripts ready | verify | 15 min |
| 5 | vLLM serve smoke (NVFP4 backend, MTP draft loads, spec decode runs) | serve config TBD | serve | 1h |
| 6 | Benchmark vs RedHat (GSM8K, MMLU-Pro) + MTP differential (tok/s with vs without spec-decode) | TBD | bench | 4-8h |
| 7 | Model card + README + HF upload | TBD | release | 2h |
| 8 | Public release (gated on user authorization) | pending | release | n/a |

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

- W4A16 GPTQ recipe → handled by sibling `pasta-paul/dsv4-flash-w4a16-fp8-mtp`
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
