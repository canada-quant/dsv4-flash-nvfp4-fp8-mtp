#!/usr/bin/env bash
# install_rtx6000pro.sh — one-shot install for Card B on RTX PRO 6000 Blackwell (SM 12.0)
#
# canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP
#
# Verified working setup as of 2026-05-26 on Brev g7e.24xlarge.
# Pin: jasl/vllm@a02a3778f on branch ds4-sm120-preview-dev.
# Patches: vllm-patches/0001 (PR #40923 MARLIN_MOE_ARCHS).
# Build time: ~30-45 min full kernel rebuild.
#
# AIME-2024 thinking-mode bench (2026-05-25): c=1=24/30, c=2=23/30, c=4=21/30
# with 0 errors and MTP draft acceptance steady at ~90.6-90.9% across all 3
# concurrency levels. c=8 PCIe-bottlenecked (use 2x TP=2 replicas instead).
set -euo pipefail

# ---- 0. Config ----
ARTIFACT="canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP"
VLLM_REPO="https://github.com/jasl/vllm.git"
VLLM_PIN="a02a3778f"
VLLM_BRANCH="ds4-sm120-preview-dev"
VLLM_SRC="${VLLM_SRC:-$HOME/src/vllm}"
VENV="${VENV:-$HOME/venv-serve}"
SCRATCH="${SCRATCH:-/scratch}"

echo "================================================================"
echo "Card B (NVFP4-MTP) RTX PRO 6000 install starting at $(date)"
echo "Pin: $VLLM_REPO @ $VLLM_PIN"
echo "Venv: $VENV"
echo "Scratch: $SCRATCH"
echo "================================================================"

# ---- 1. Scratch + HF cache on fast disk ----
if [ ! -L "$SCRATCH" ] && [ ! -d "$SCRATCH" ]; then
    if [ -d /opt/dlami/nvme ]; then
        sudo ln -sfn /opt/dlami/nvme "$SCRATCH" && sudo chown -h "$USER:$USER" "$SCRATCH"
    else
        sudo mkdir -p "$SCRATCH" && sudo chown "$USER:$USER" "$SCRATCH"
    fi
fi
export HF_HOME="$SCRATCH/hf-cache"
mkdir -p "$HF_HOME"
mkdir -p "$HOME/src"

# ---- 2. CUDA toolkit ----
# DLAMI ships an incomplete /opt/pytorch/cuda. Use system /usr/local/cuda.
if [ ! -f /usr/local/cuda/lib64/libcudart.so ]; then
    echo "[step] installing cuda-toolkit-13-0..."
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-13-0
fi
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# ---- 3. Python 3.10 venv ----
if [ ! -d "$VENV" ]; then
    echo "[step] creating Python 3.10 venv..."
    python3.10 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# ---- 4. Clone + pin jasl/vllm ----
if [ ! -d "$VLLM_SRC/.git" ]; then
    echo "[step] cloning $VLLM_REPO..."
    git clone "$VLLM_REPO" "$VLLM_SRC"
fi
cd "$VLLM_SRC"
git fetch origin "$VLLM_BRANCH"
git checkout "$VLLM_PIN"
echo "[step] vLLM pinned at $(git rev-parse --short HEAD)"

# ---- 5. Apply vllm-patches/ ----
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
# Detect: are we running INSIDE the canada-quant repo or did we curl-pipe?
if [ -f "$REPO_ROOT/vllm-patches/0001_marlin_moe_archs_40923.patch" ]; then
    PATCHES="$REPO_ROOT/vllm-patches"
else
    # Pull patches from canada-quant/dsv4-flash-nvfp4-fp8-mtp main
    PATCHES_TMPDIR="$(mktemp -d)"
    PATCHES="$PATCHES_TMPDIR/vllm-patches"
    mkdir -p "$PATCHES"
    PATCH_RAW="https://raw.githubusercontent.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/main/vllm-patches"
    for p in 0001_marlin_moe_archs_40923 0002_marlin_moe_workspace_4x 0003_marlin_moe_c_tmp_36889; do
        curl -sL "$PATCH_RAW/${p}.patch" -o "$PATCHES/${p}.patch"
    done
fi

cd "$VLLM_SRC"
for p in "$PATCHES"/0001_*.patch "$PATCHES"/0003_*.patch; do
    # 0002 is a Python-side single-line sed (workspace 4x); applied separately below
    [ -f "$p" ] || continue
    echo "[patch] applying $(basename "$p")..."
    if git apply --check "$p" 2>/dev/null; then
        git apply "$p"
    else
        echo "[patch]   already present, skipping"
    fi
done

# 0002: workspace 4x oversize via sed (idempotent)
echo "[patch] applying workspace 4x (vllm/model_executor/layers/quantization/utils/marlin_utils.py)..."
WS_FILE="vllm/model_executor/layers/quantization/utils/marlin_utils.py"
if ! grep -q "max_blocks_per_sm \* 4" "$WS_FILE"; then
    sed -i 's/sms \* max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False/sms * max_blocks_per_sm * 4, dtype=torch.int, device=device, requires_grad=False/' "$WS_FILE"
    echo "[patch]   applied"
else
    echo "[patch]   already present"
fi

# ---- 6. Build vLLM ----
export TORCH_CUDA_ARCH_LIST="12.0a"
export MAX_JOBS=12
export CMAKE_BUILD_PARALLEL_LEVEL=12

echo "[build] pip install -e . (~30-45 min)..."
# The install can crash at the spinloop.abi3.so install step (DLAMI Python 3.10
# gotcha) — but _moe_C.abi3.so will have been produced. We tolerate the install
# crash and manually copy the .so afterward.
set +e
pip install --no-deps --no-build-isolation -v -e . 2>&1 | tee /tmp/vllm_build.log | tail -40
BUILD_RC=$?
set -e

# ---- 7. Manual _moe_C.abi3.so copy if needed ----
SITE_VLLM="$(python -c 'import sys; import os; [print(os.path.dirname(p)+"/vllm") for p in sys.path if os.path.exists(p+"/vllm/__init__.py")][0:1]' 2>/dev/null || true)"
SITE_VLLM="${SITE_VLLM:-$VENV/lib/python3.10/site-packages/vllm}"
if [ ! -f "$SITE_VLLM/_moe_C.abi3.so" ] || [ "$VLLM_SRC/vllm/_moe_C.abi3.so" -nt "$SITE_VLLM/_moe_C.abi3.so" ]; then
    echo "[build] copying _moe_C.abi3.so to $SITE_VLLM..."
    cp -v "$VLLM_SRC/vllm/_moe_C.abi3.so" "$SITE_VLLM/_moe_C.abi3.so"
fi

# ---- 8. Runtime dependencies ----
echo "[step] installing runtime deps..."
pip install --upgrade huggingface_hub hf-transfer langdetect immutabledict nltk evalplus openai datasets sentencepiece

# ---- 9. Download artifact ----
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "[step] downloading $ARTIFACT (172 GB; ~5-10 min on hf-transfer)..."
hf download "$ARTIFACT"

# ---- 10. Smoke + summary ----
echo
echo "================================================================"
echo "Install complete at $(date)"
echo "================================================================"
echo
echo "  CUDA_HOME:    $CUDA_HOME"
echo "  Venv:         $VENV"
echo "  vLLM pin:     $(cd $VLLM_SRC && git rev-parse --short HEAD)"
echo "  Patches:      0001 (MARLIN_MOE_ARCHS), 0002 (workspace 4x), 0003 (c_tmp)"
echo "  Artifact:     $ARTIFACT"
echo
echo "To serve (TP=4):"
echo
echo "  source $VENV/bin/activate"
echo "  CUDA_HOME=/usr/local/cuda \\"
echo "  TORCH_CUDA_ARCH_LIST=12.0a \\"
echo "  VLLM_TEST_FORCE_FP8_MARLIN=1 \\"
echo "  vllm serve $ARTIFACT \\"
echo "    --tensor-parallel-size 4 \\"
echo "    --kv-cache-dtype fp8 --block-size 256 \\"
echo "    --max-model-len 16384 \\"
echo "    --max-num-seqs 8 --max-num-batched-tokens 8192 \\"
echo "    --gpu-memory-utilization 0.95 \\"
echo "    --compilation-config '{\"max_cudagraph_capture_size\":8,\"cudagraph_capture_sizes\":[1,2,4,8]}' \\"
echo "    --no-enable-prefix-caching \\"
echo "    --tokenizer-mode deepseek_v4 \\"
echo "    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \\"
echo "    --reasoning-parser deepseek_v4 \\"
echo "    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}' \\"
echo "    --disable-custom-all-reduce \\"
echo "    --trust-remote-code"
echo
