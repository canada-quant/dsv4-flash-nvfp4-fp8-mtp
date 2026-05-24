#!/usr/bin/env bash
# Serve the NVFP4+FP8+MTP artifact on RTX PRO 6000 Blackwell (SM 12.0).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1     bash scripts/serve_rtx6000pro.sh <model_path> <port> 2
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/serve_rtx6000pro.sh <model_path> <port> 4
#
# Notes:
#   - SM 12.0 = Blackwell consumer/server (RTX PRO 6000); B300 datacenter is
#     SM 10.3. Build vLLM with TORCH_CUDA_ARCH_LIST=12.0a.
#   - --disable-custom-all-reduce: RTX 6000 Pro has no NVLink. Custom AR
#     crashes with CUDA invalid-argument; PCIe NCCL is the right path.
#   - --speculative-config num_speculative_tokens=1: DeepGemm next_n
#     assertion on Hopper + SM12 attention.hpp paths also assert next_n==1.
#     k=1 is the only viable spec config here.
#   - VLLM_TEST_FORCE_FP8_MARLIN=1: NVFP4 SM12 ops on attention's FP8_BLOCK
#     128x128 path go through Marlin until DeepGemm sm120 catches up.

set -euo pipefail

MODEL_PATH="${1:?usage: $0 <model_path> <port> <tp>}"
PORT="${2:?usage: $0 <model_path> <port> <tp>}"
TP="${3:?usage: $0 <model_path> <port> <tp>}"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0a"
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# VLLM_TEST_FORCE_FP8_MARLIN=1 forces MoE NVFP4 to take the Marlin backend.
# The default NVFP4 selector filters out everything except FLASHINFER_TRTLLM
# when the model has swiglu_limit set (DSV4-Flash sets swiglu_limit=10.0),
# and FLASHINFER_TRTLLM doesn't support SM 12.0. With Marlin forced, MoE
# works. The companion patch (skip-marlin-for-bmm) prevents wo_a/wo_b from
# being repacked into Marlin tile layout — those layers use the SM12 Triton
# fp8_einsum kernel directly.
export VLLM_TEST_FORCE_FP8_MARLIN=1

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "$HOME/venv-serve/bin/activate" ]]; then
    source "$HOME/venv-serve/bin/activate"
fi

exec vllm serve "$MODEL_PATH" \
    --served-model-name DSV4-NVFP4-FP8-MTP deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash \
    --tensor-parallel-size "$TP" \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 4096 \
    --max-num-seqs 8 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.97 \
    --compilation-config '{"max_cudagraph_capture_size": 16, "cudagraph_capture_sizes": [1, 2, 4, 8, 16]}' \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --disable-custom-all-reduce \
    --trust-remote-code --host 0.0.0.0 --port "$PORT"
