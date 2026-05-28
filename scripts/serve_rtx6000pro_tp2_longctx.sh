#!/usr/bin/env bash
# serve_rtx6000pro_tp2_longctx.sh — Card B serve on 2× RTX PRO 6000 with MAX
# context window at max_num_seqs=1 (single-user). MTP still enabled.
#
# Memory math at TP=2 + FP8 KV cache + compressed MLA:
#   Weights:        86 GB/GPU
#   MTP block:       6.5 GB/GPU
#   Free for KV+graph+activations: ~3.5 GB/GPU
#   Per-token KV (after TP=2 split): ~16 KB
#   So 128K context = 2 GB/GPU KV — tight but should fit.
#
# Override with MAX_MODEL_LEN=<n> to try different ceilings.

set -uo pipefail

MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP}"
MODEL_NAME="${MODEL_NAME:-DSV4-NVFP4-FP8-MTP}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS=1

echo "[serve] long-ctx Card B on 2× RTX PRO 6000 TP=2 with MTP"
echo "[serve]   max_model_len: $MAX_MODEL_LEN   max_num_seqs: $MAX_NUM_SEQS"

export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_USE_LAYERNAME=0
export HF_HUB_ENABLE_HF_TRANSFER=1

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_capture_sizes":[2],"max_cudagraph_capture_size":2}' \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
