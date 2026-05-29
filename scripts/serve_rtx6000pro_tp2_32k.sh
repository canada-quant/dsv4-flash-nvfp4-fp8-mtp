#!/usr/bin/env bash
# serve_rtx6000pro_tp2_32k.sh — Card B serve on 2× RTX PRO 6000 with 32K context
# at max_num_seqs=1 (single-user). MTP enabled. CUDA graphs ON.
#
# Sized to give AIME thinking-mode enough budget (32K) that 0 problems
# truncate — apples-apples vs documented B300 baseline.

set -uo pipefail

MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP}"
MODEL_NAME="${MODEL_NAME:-DSV4-NVFP4-FP8-MTP}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS=1

echo "[serve] Card B on 2× RTX PRO 6000 TP=2 with MTP — 32K context"

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
