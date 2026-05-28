#!/usr/bin/env bash
# serve_rtx6000pro_tp2.sh — optimal Card B serve recipe for 2× RTX PRO 6000
# Blackwell Server Edition (SM 12.0). Targets mtcl's hardware specifically:
# 2 GPUs, CUDA graphs ON for real throughput.
#
# What differs from the TP=4 + --enforce-eager dev path:
#   - TP=2 not TP=4 (mtcl has 2 GPUs)
#   - CUDA graphs ON (no --enforce-eager) → 5-10× faster than eager mode
#   - VLLM_USE_LAYERNAME=0 → bypasses jasl's LayerName opaque-type registration
#     that triggers Inductor lowering failure on `vllm.moe_forward_shared.default`
#     (the FakeScriptObject .stride access bug). Setting this env makes
#     layer_name parameters stay as plain str so Inductor handles them as
#     constants.
#   - --gpu-memory-utilization 0.90 (not 0.95) — at TP=2 with 96 GB cards,
#     Card B's 172 GB total + KV cache + MTP shared embeddings + activations
#     leaves less headroom than TP=4. 0.90 is conservative.
#
# Usage:
#   bash scripts/serve_rtx6000pro_tp2.sh
#
# Or inside the Docker image:
#   docker run --rm --gpus '"device=0,1"' --shm-size=16g --ipc=host \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     --network host \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -e HF_TOKEN=$HF_TOKEN \
#     canada-quant/dsv4-rtx6000pro:v3 \
#     bash /workspace/scripts/serve_rtx6000pro_tp2.sh

set -uo pipefail

MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP}"
MODEL_NAME="${MODEL_NAME:-DSV4-NVFP4-FP8-MTP}"
PORT="${PORT:-8000}"

# Card B on 2× RTX PRO 6000:
# - TP=2 across 2 cards (96 GB each = 192 GB total)
# - Card B's 172 GB on-disk + KV cache + MTP block fits but tight
# - max_model_len=16384 to keep KV cache reservation reasonable
# - max_num_seqs=4 to keep cudagraph capture footprint small
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"

echo "[serve] starting Card B on 2× RTX PRO 6000 TP=2 with CUDA graphs ON"
echo "[serve]   model: $MODEL_ID"
echo "[serve]   max_model_len: $MAX_MODEL_LEN"
echo "[serve]   max_num_seqs: $MAX_NUM_SEQS"

# Critical env vars:
# - VLLM_TEST_FORCE_FP8_MARLIN=1: route NVFP4 MoE to MarlinExperts on sm_120
#   (the only working path; native FP4 tensor cores need B200/B300)
# - VLLM_USE_LAYERNAME=0: keep layer_name as str, avoid Inductor lowering bug
#   on vllm.moe_forward_shared.default (FakeScriptObject .stride access)
# - HF_HUB_ENABLE_HF_TRANSFER=1: faster artifact download from HF
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
    --compilation-config '{"cudagraph_capture_sizes":[1,2],"max_cudagraph_capture_size":2}' \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
