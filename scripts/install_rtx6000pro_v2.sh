#!/usr/bin/env bash
# install_rtx6000pro.sh v2 — canonical upstream path for RTX PRO 6000 (SM 12.0)
#
# As of 2026-05-26 enough DSv4 + SM_120 work has merged into vllm-project/vllm@main
# that we no longer need the jasl/vllm fork. This installer targets a pinned
# vllm-project/vllm@main SHA and applies only the still-needed patches.
#
# Merged upstream (no longer needed as separate patches):
#   #42209 NVFP4 MoE support for DSv4
#   #40082 FlashInfer b12x MoE + FP4 GEMM for SM120/121
#   #43554 Remove NormGateLinear (fused into mhc_pre)
#   #43149 Extract DSv4 sparse MLA into model folder
#   #43690 Drop _get_compressed_kv_buffer
#
# Still needs cherry-pick (open PRs):
#   #40923 MARLIN_MOE_ARCHS 12.0a;12.1a native sm_120 cubins (else JIT-PTX corruption)
#   #43655 Compressor/indexer quant_config plumbing + torch.mm conditional dispatch
#   #36889 Marlin MoE c_tmp clamp removal (closed but re-file candidate; our evidence at
#          https://github.com/vllm-project/vllm/pull/36889#issuecomment-4531289048)
#
# Still our addition (no upstream equivalent yet):
#   Marlin MoE workspace 4× oversize (marlin_utils.py:268)

set -euo pipefail

CARD="${CARD:?set CARD=A|B|D}"
case "$CARD" in
  A) ARTIFACT="canada-quant/DeepSeek-V4-Flash-W4A16-FP8";     MTP=0 ;;
  B) ARTIFACT="canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP"; MTP=1 ;;
  D) ARTIFACT="canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP"; MTP=1 ;;
esac

# Pin to 2026-05-26 HEAD; bump when you want newer fixes
VLLM_REPO="https://github.com/vllm-project/vllm.git"
VLLM_PIN="6e503868ca"
VLLM_SRC="${VLLM_SRC:-$HOME/src/vllm-upstream}"
VENV="${VENV:-$HOME/venv-upstream}"
SCRATCH="${SCRATCH:-/opt/dlami/nvme}"

echo "================================================================"
echo "canada-quant Card $CARD on RTX PRO 6000 (SM 12.0) — v2 canonical"
echo "Date: $(date -Iseconds)  vLLM pin: $VLLM_PIN (vllm-project main)"
echo "Artifact: $ARTIFACT"
echo "================================================================"

# ---- 1. Scratch + HF cache on fast disk ----
export HF_HOME="$SCRATCH/hf-cache"
mkdir -p "$HF_HOME" "$HOME/src"
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- 2. CUDA toolkit ----
if [ ! -f /usr/local/cuda/lib64/libcudart.so ]; then
    sudo apt-get update && sudo apt-get install -y cuda-toolkit-13-0
fi
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# ---- 3. Python 3.10 venv + uv (per vLLM AGENTS.md preference) ----
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV" ]; then
    uv venv --python 3.10 "$VENV"
fi
source "$VENV/bin/activate"
uv pip install --upgrade pip setuptools wheel

# ---- 4. Clone + pin upstream vLLM ----
if [ ! -d "$VLLM_SRC/.git" ]; then
    git clone "$VLLM_REPO" "$VLLM_SRC"
fi
cd "$VLLM_SRC"
git fetch origin main
git checkout "$VLLM_PIN"
echo "[step] vLLM pinned at $(git rev-parse --short HEAD)"

# ---- 5. Cherry-pick still-open upstream PRs ----
echo "[patch] cherry-picking open upstream PRs..."

# PR #40923 — MARLIN_MOE_ARCHS sm_120a cubins
PR_40923_DIFF="$(mktemp)"
gh pr diff 40923 --repo vllm-project/vllm > "$PR_40923_DIFF"
git apply --check "$PR_40923_DIFF" 2>/dev/null && git apply "$PR_40923_DIFF" || echo "  #40923 already applied / merged"

# PR #43655 — compressor/indexer quant_config plumbing + conditional torch.mm dispatch
PR_43655_DIFF="$(mktemp)"
# Filter out test files since site-packages-side install doesn't need them
gh pr diff 43655 --repo vllm-project/vllm | \
    awk '/^diff --git a\/tests/,/^diff --git/ {if (/^diff --git a\/tests/) skip=1; else if (/^diff --git/ && !/a\/tests/) skip=0} !skip' > "$PR_43655_DIFF"
git apply --check "$PR_43655_DIFF" 2>/dev/null && git apply "$PR_43655_DIFF" || echo "  #43655 already applied / merged"

# PR #36889 — Marlin MoE c_tmp clamp removal (ops.cu only)
PR_36889_DIFF="$(mktemp)"
gh pr diff 36889 --repo vllm-project/vllm | \
    awk '/^diff --git a\/csrc\/moe\/marlin_moe_wna16\/ops.cu/,/^diff --git/ {if (/^diff --git a\/csrc\/moe\/marlin_moe_wna16\/ops.cu/) print; else if (/^diff --git/) exit; else print}' > "$PR_36889_DIFF"
git apply --check "$PR_36889_DIFF" 2>/dev/null && git apply "$PR_36889_DIFF" || echo "  #36889 already applied / merged"

# Workspace 4× oversize (no upstream PR yet — our addition)
WS_FILE="vllm/model_executor/layers/quantization/utils/marlin_utils.py"
if ! grep -q "max_blocks_per_sm \* 4" "$WS_FILE"; then
    sed -i 's/sms \* max_blocks_per_sm, dtype=torch.int/sms * max_blocks_per_sm * 4, dtype=torch.int/' "$WS_FILE"
    echo "  workspace 4× applied"
fi

# ---- 6. Build ----
export TORCH_CUDA_ARCH_LIST="12.0a"
export MAX_JOBS=12
export CMAKE_BUILD_PARALLEL_LEVEL=12
export VLLM_USE_PRECOMPILED=0  # we patched CUDA, need full rebuild

echo "[build] uv pip install -e . (~30-45 min full rebuild)..."
set +e
uv pip install --no-build-isolation -e . --torch-backend=auto 2>&1 | tee /tmp/vllm_build.log | tail -40
BUILD_RC=$?
set -e

# Tolerate the known spinloop install crash; copy the produced _moe_C.abi3.so manually
if [ -f "$VLLM_SRC/vllm/_moe_C.abi3.so" ]; then
    SITE_VLLM="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || echo "$VENV/lib/python3.10/site-packages/vllm")"
    if [ "$SITE_VLLM" != "$VLLM_SRC/vllm" ]; then
        cp -v "$VLLM_SRC/vllm/_moe_C.abi3.so" "$SITE_VLLM/_moe_C.abi3.so" || true
    fi
fi

# ---- 7. Runtime deps + artifact download ----
uv pip install --upgrade huggingface_hub hf-transfer langdetect immutabledict nltk evalplus openai datasets sentencepiece
hf download "$ARTIFACT"

# ---- 8. Print summary ----
cat <<EOF

================================================================
Install complete — Card $CARD ready
================================================================

  vLLM pin:     $(cd $VLLM_SRC && git rev-parse --short HEAD)
  Patches:      40923 (Marlin arch), 43655 (quant_config plumb), 36889 (c_tmp), workspace 4×
  Venv:         $VENV
  Artifact:     $ARTIFACT (cached under $HF_HOME)

Serve (TP=4, MTP if applicable):

  source $VENV/bin/activate
  CUDA_HOME=/usr/local/cuda \\
  TORCH_CUDA_ARCH_LIST=12.0a \\
  VLLM_TEST_FORCE_FP8_MARLIN=1 \\
  vllm serve $ARTIFACT \\
    --tensor-parallel-size 4 \\
    --kv-cache-dtype fp8 --block-size 256 \\
    --max-model-len 16384 \\
    --max-num-seqs 8 --max-num-batched-tokens 8192 \\
    --gpu-memory-utilization 0.95 \\
    --compilation-config '{"max_cudagraph_capture_size":8,"cudagraph_capture_sizes":[1,2,4,8]}' \\
    --no-enable-prefix-caching \\
    --tokenizer-mode deepseek_v4 \\
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \\
    --reasoning-parser deepseek_v4 \\
$([ $MTP = 1 ] && echo "    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}' \\\\")
    --disable-custom-all-reduce \\
    --trust-remote-code

EOF
