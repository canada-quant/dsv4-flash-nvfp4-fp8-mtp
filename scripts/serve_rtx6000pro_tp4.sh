#!/usr/bin/env bash
# serve_rtx6000pro_tp4.sh — Card B serve on 4× RTX PRO 6000 Blackwell Server
# Edition (SM 12.0). For dual-card mtcl-style setups, use serve_rtx6000pro_tp2.sh
# instead — this is the higher-headroom variant for 4-GPU rigs.
#
# Differs from TP=2 variant:
#   - 4× more GPU memory total → larger max_model_len (16K vs 8K)
#   - More cudagraph capture sizes for concurrency
#   - Same patches, same env vars
#
# Usage (inside docker):
#   docker run --rm --gpus all --shm-size=16g --ipc=host \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     --network host \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v $(pwd)/scripts:/workspace/scripts:ro \
#     canada-quant/dsv4-rtx6000pro:v3 \
#     bash /workspace/scripts/serve_rtx6000pro_tp4.sh

set -uo pipefail

MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP}"
MODEL_NAME="${MODEL_NAME:-DSV4-NVFP4-FP8-MTP}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

echo "[serve] starting Card B on 4× RTX PRO 6000 TP=4 with CUDA graphs ON"
echo "[serve]   model: $MODEL_ID"
echo "[serve]   max_model_len: $MAX_MODEL_LEN  max_num_seqs: $MAX_NUM_SEQS"

export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_USE_LAYERNAME=0
export HF_HUB_ENABLE_HF_TRANSFER=1

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
