# Phase 7 — HF upload prep

**This document is gated on user authorization per `CLAUDE.md`. Do not execute steps below until the user explicitly says "go" or "ship" or "upload."**

## Target

- HF repo: `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` (create as **public** once auth'd; ATM the GitHub source repo is private)
- License: inherits upstream DSV4-Flash license (whatever DeepSeek shipped)
- Tags: `compressed-tensors`, `nvfp4`, `vllm`, `deepseek`, `mtp`, `speculative-decoding`

## What ships

From `/scratch/weights/v4-flash-nvfp4-fp8-mtp/` on the B300 box:

| File | Size | Notes |
|---|---|---|
| `model.safetensors.index.json` | ~5 MB | Final index |
| `model-{1..35}-of-35.safetensors` | ~172 GB total | 35 shards |
| `config.json` | ~7 KB | Final config — FUSED attn regex + W8A8 input_activations + `scale_fmt: ue8m0` + layer-43 + mtp_block in ignore list |
| `tokenizer.json`, `tokenizer_config.json` | from upstream | DSV4-Flash tokenizer |
| `generation_config.json` | from upstream | DSV4 defaults |
| `MODEL_CARD.md` | this repo | as `README.md` on HF |

**Do NOT upload**:
- `_progress.json` (calibration internal state)
- `config.json.bak_*` files (backups from postprocess steps) — there are 7 of these; delete or .gitignore
- `_per_layer_checkpoints/*` if present (calibration intermediate state)

## Pre-upload checklist

```bash
# 1. Verify artifact integrity
cd /scratch/weights/v4-flash-nvfp4-fp8-mtp
python /home/paul/dsv4-flash-nvfp4-fp8-mtp/scripts/verify_mtp_quantized.py .
# Expect: "MTP gate PASSED — block present, bfloat16, unquantized." + 799 MTP weight tensors

# 2. Verify config.json has all the post-processed fields
python -c "
import json
c = json.load(open('config.json'))
qc = c['quantization_config']
assert qc['scale_fmt'] == 'ue8m0', 'scale_fmt missing/wrong'
assert qc['format'] == 'mixed-precision', 'format wrong'
g0 = qc['config_groups']['group_0']
assert 'fused_wqa_wkv' in str(g0['targets']), 'group_0 not using fused names'
assert g0['input_activations'] is not None, 'group_0 missing input_activations'
assert g0['input_activations']['dynamic'] == True, 'group_0 input_activations not dynamic'
assert c['num_hidden_layers'] == 43, 'num_hidden_layers wrong'
assert c['num_nextn_predict_layers'] == 1, 'num_nextn_predict_layers wrong'
assert c['expert_dtype'] == 'fp4', 'expert_dtype wrong'
print('config OK')
"

# 3. Clean backup files (NOT atomic — only run when sure)
ls config.json.bak_* 2>/dev/null  # review first
rm -f config.json.bak_*           # then delete

# 4. Check no extra files
ls -la | grep -vE '\.safetensors$|\.safetensors\.index\.json$|\.json$|README.md|\.md$' | head -5

# 5. (Optional) re-run lm-eval GSM8K smoke to confirm artifact still serves
# vllm serve must be up at port 8089:
lm_eval run --model local-chat-completions --tasks gsm8k --num_fewshot 8 --limit 50 \
  --model_args "model=$ARTIFACT_DIR,base_url=http://localhost:8089/v1/chat/completions" \
  --apply_chat_template --batch_size 1
```

## Upload commands (DO NOT EXECUTE WITHOUT AUTH)

```bash
# Login (token must have write access to canada-quant org)
huggingface-cli login

# Create the repo (public)
huggingface-cli repo create DeepSeek-V4-Flash-NVFP4-FP8-MTP --type model --organization canada-quant

# Upload artifact files (use --commit-message for clear history)
cd /scratch/weights/v4-flash-nvfp4-fp8-mtp
huggingface-cli upload canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP . . \
    --include "*.safetensors" "*.json" \
    --exclude "_progress.json" "*.bak_*" "_per_layer_checkpoints/*" \
    --commit-message "Initial NVFP4-FP8-MTP artifact upload"

# Upload model card as README
cp /home/paul/dsv4-flash-nvfp4-fp8-mtp/MODEL_CARD.md /tmp/README.md
huggingface-cli upload canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    /tmp/README.md README.md \
    --commit-message "Model card with GSM8K 0.918/0.952 + MMLU-Pro 0.811"
```

## Post-upload sanity

```bash
# Verify upload visible
curl -sL "https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP/raw/main/config.json" \
    | python3 -c "import json, sys; c=json.load(sys.stdin); print('arch:', c['architectures']); print('mtp present:', c.get('num_nextn_predict_layers', 0) > 0)"

# Sanity-load from HF via vLLM (TP=4 on a B300 box)
vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --tensor-parallel-size 4 --port 8089 --kv-cache-dtype fp8

# Test completion
curl -X POST http://localhost:8089/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP", "prompt": "The capital of France is", "max_tokens": 10, "temperature": 0}'
# Expect: "Paris."
```

## After upload — announcement / outreach

- Comment on each of our upstream PRs (#43248, #43288, #43290) linking to the artifact as a real-world reproducer / use case
- Comment on llm-compressor #2745 + vLLM #43297 + #43304 linking to the artifact as the motivating example
- (Optional) social: post about the MTP-preserving DSV4-Flash NVFP4-FP8 quant with measured numbers

## What NOT to claim

- DO NOT claim "spec-decode acceptance rate X%" — Phase 5b is currently gated on upstream fixes. The model card already states this honestly.
- DO NOT claim "matches BF16 perfectly" on GSM8K — the flexible-extract match is metric-specific; strict-match shows 3.4 pts drop vs BF16.
- DO NOT claim DeepSeek Pro support; this artifact targets DeepSeek-V4-Flash only.

## After artifact ships

Track upstream PRs/issues for adoption:
- vLLM PR #43248, #43288, #43290 (this org) — defensive code-smell fixes; likely merge over weeks
- vLLM issue #43297 — global_scale loader broadcast; expect a fix PR from maintainers
- vLLM issue #43304 — MTP draft model quant_config inheritance; expect a more substantive fix
- llm-compressor #2745 — MTP inference-mode crash; the root upstream issue gating Phase 5b

When any of these merge, re-run Phase 5b with `--speculative-config method=mtp` to fill in the measured acceptance-rate number on the model card.
