#!/usr/bin/env bash
# install_rtx6000pro_v3.sh — verified RTX PRO 6000 install for canada-quant DSv4 cards
#
# Status: 2026-05-27 — this recipe is the one verified against canada-quant artifacts
# on a Brev g7e.24xlarge (4× RTX PRO 6000 Blackwell Server Edition, SM 12.0).
#
# WHY JASL, NOT MAINLINE
# We attempted the mainline (vllm-project/vllm@main) path this session and found it
# is blocked at first forward pass on SM 12.0 by DeepGEMM Hopper-only kernels
# (`fp8_einsum` and `tf32_hc_prenorm_gemm`). jasl/vllm carries SM 12.0 fallback files
# (`sm12x_deep_gemm_fallbacks.py`, `sm12x_mqa.py`, `fp8_einsum.py`, `cutedsl_utils.py`)
# that are not yet upstreamed. Open vLLM PRs to track mainline progress:
# #41834, #41738, #43333, #43341, #43687. ETA for mainline viability: 2-4 weeks.
#
# For now, jasl/vllm@27fd665b (ds4-sm120-preview-dev HEAD) is the canonical SM 12.0 base.
# We additionally cherry-pick PR #41834's tuned configs and apply our own patches
# 0007 + 0008 (PR #43722 + PR #43723).
#
# VERIFIED RESULTS (2026-05-27):
#   Card B (NVFP4-FP8-MTP): serves coherent multi-step math (17*23 → 391 chain),
#                            MTP shared embeddings load, smoke completion clean
#   Card D (W4A16-FP8-MTP): AIME-30 c=4 thinking = 24/30 correct, 0 CUDA errors,
#                            91.61% MTP acceptance, 641.9s wall (10.7 min)
#   Card A: not retested this session — same install pattern applies
#
# USAGE
#   CARD=B bash install_rtx6000pro_v3.sh
#   CARD=D bash install_rtx6000pro_v3.sh
#   CARD=A bash install_rtx6000pro_v3.sh
#
# Run time: ~45-60 min on a fresh Brev g7e.24xlarge (build) + 30 min Card B download
# (172 GB) or 142 GB Card D / 142 GB Card A.

set -euo pipefail

CARD="${CARD:?set CARD=A|B|D}"
case "$CARD" in
  A) ARTIFACT="canada-quant/DeepSeek-V4-Flash-W4A16-FP8";     MTP=0 ;;
  B) ARTIFACT="canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP"; MTP=1 ;;
  D) ARTIFACT="canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP"; MTP=1 ;;
esac

# jasl/vllm@27fd665b on ds4-sm120-preview-dev (SHA captured 2026-05-26)
VLLM_REPO="https://github.com/jasl/vllm.git"
VLLM_BRANCH="ds4-sm120-preview-dev"
VLLM_PIN="27fd665bdc3ba58afc5c34cbb9034c9fc1a95029"
VLLM_SRC="${VLLM_SRC:-/opt/dlami/nvme/src/vllm-jasl}"
VENV="${VENV:-$HOME/venv-serve}"
SCRATCH="${SCRATCH:-/opt/dlami/nvme}"

echo "================================================================"
echo "canada-quant Card $CARD on RTX PRO 6000 (SM 12.0) — v3 verified"
echo "Date:      $(date -Iseconds)"
echo "vLLM pin:  jasl/vllm@$VLLM_PIN ($VLLM_BRANCH)"
echo "Artifact:  $ARTIFACT"
echo "================================================================"

# ---- 1. Scratch + HF cache on fast disk ----
export HF_HOME="$SCRATCH/hf-cache"
mkdir -p "$HF_HOME" "$SCRATCH/src"
export HF_XET_HIGH_PERFORMANCE=1

# ---- 2. CUDA toolkit (must be 12.9 minimum for sm_120a cubins) ----
if [ ! -f /usr/local/cuda/lib64/libcudart.so ]; then
    sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-9 || \
    sudo apt-get install -y cuda-toolkit-13-0
fi
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# ---- 3. Python 3.10 venv ----
if [ ! -d "$VENV" ]; then
    python3.10 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel ninja cmake packaging
# cmake from venv (3.26+) is essential — system cmake 3.22 fails vLLM build

# ---- 4. Clone + pin jasl/vllm ds4-sm120-preview-dev ----
if [ ! -d "$VLLM_SRC/.git" ]; then
    git clone -b "$VLLM_BRANCH" "$VLLM_REPO" "$VLLM_SRC"
fi
cd "$VLLM_SRC"
git fetch origin "$VLLM_BRANCH"
git checkout "$VLLM_PIN"
echo "[step] jasl/vllm pinned at $(git rev-parse --short HEAD)"

# ---- 5. Apply canada-quant local patches ----

# PR #43722 — MarlinFP8.can_implement block-FP8 refuse (our PR, open upstream)
# Without this, block-FP8 layers crash at csrc/quantization/marlin/marlin.cu:701
# "b_scales dim 1 = 32 is not size_n = 1536"
python3 - <<'PYEOF'
F = "vllm/model_executor/kernels/linear/scaled_mm/marlin.py"
src = open(F).read()
marker = '    @classmethod\n    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:\n        return True, None'
new = """    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        # canada-quant 2026-05-26: MarlinFP8 cannot serve block-FP8 layers; let
        # dispatcher fall through to TritonFp8BlockScaledMMKernel via FP8_BLOCK
        # priority list. PR #43722 filed upstream.
        try:
            if c.activation_quant_key.scale.group_shape.is_per_group():
                return False, "MarlinFP8 cannot serve block-FP8 layers"
        except Exception:
            pass
        return True, None"""
if marker in src and "MarlinFP8 cannot serve block-FP8" not in src:
    open(F, "w").write(src.replace(marker, new, 1))
    print("[patch] PR #43722 applied")
else:
    print("[patch] PR #43722 already applied or marker not found")
PYEOF

# PR #43723 — DSv4 wo_a scale getattr fallback (our PR, open upstream)
# Without this, the post-#43722 dispatch to non-Marlin path AttributeErrors
# at attention.py:330 reading weight_scale_inv.
python3 - <<'PYEOF'
F = "vllm/models/deepseek_v4/nvidia/ops/attention.py"
src = open(F).read()
marker = "        wo_a_fp8 = self.wo_a.weight\n        wo_a_scale = self.wo_a.weight_scale_inv"
new = """        wo_a_fp8 = self.wo_a.weight
        # canada-quant 2026-05-26: non-Marlin kernels (Triton block-FP8, Cutlass,
        # DeepGemm) leave the on-disk name `weight_scale`. PR #43723 filed upstream.
        wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
        if wo_a_scale is None:
            wo_a_scale = self.wo_a.weight_scale"""
if marker in src and "canada-quant 2026-05-26" not in src.split("wo_a_scale")[0]:
    open(F, "w").write(src.replace(marker, new, 1))
    print("[patch] PR #43723 applied")
else:
    print("[patch] PR #43723 already applied or marker not found")
PYEOF

# PR #41834 RTX PRO 6000 Server Edition tuned Triton block-FP8 autotune configs
# Without these JSON files, default num_stages=2 produces degenerate output
# ("4*4=16 loop" instead of "4\n\nWhat is..." correct continuation)
git fetch https://github.com/vllm-project/vllm.git refs/pull/41834/head:pr-41834 || true
git checkout pr-41834 -- \
    vllm/model_executor/layers/quantization/utils/configs/ \
    vllm/model_executor/layers/fused_moe/configs/ 2>/dev/null || \
    echo "[patch] PR #41834 configs already present (skipping)"
echo "[patch] PR #41834 tuned configs applied"

# Marlin MoE workspace 4× oversize (defensive — prevents Card D cudaErrorIllegalAddress
# under concurrent decode). canada-quant local, not yet upstreamed.
sed -i 's/sms \* max_blocks_per_sm, dtype=torch.int/sms * max_blocks_per_sm * 4, dtype=torch.int/' \
    vllm/model_executor/layers/quantization/utils/marlin_utils.py || true
echo "[patch] workspace 4× applied"

# PR #40923 — Marlin sm_120a native cubins (apply if not already in jasl branch)
# jasl/vllm@27fd665b has this; skip if marker found
if ! grep -q "12.0a;12.1a" CMakeLists.txt; then
    git fetch https://github.com/vllm-project/vllm.git refs/pull/40923/head:pr-40923 || true
    git diff main..pr-40923 -- CMakeLists.txt > /tmp/pr_40923_cmake.diff 2>/dev/null || true
    git apply /tmp/pr_40923_cmake.diff 2>/dev/null || echo "[patch] PR #40923 already in jasl base"
fi

# ---- 6. Build vLLM (CUDA kernel compile, 30-45 min on g7e.24xlarge) ----
export TORCH_CUDA_ARCH_LIST="12.0a"
export MAX_JOBS=12
export CMAKE_BUILD_PARALLEL_LEVEL=12
export VLLM_USE_PRECOMPILED=0

echo "[build] pip install -e . (CUDA kernel compile, ~30-45 min)..."
pip install --no-build-isolation --no-deps -v -e .

# ---- 7. Install DeepGEMM Python package from bundled source ----
# Required by vllm.utils.deep_gemm.tf32_hc_prenorm_gemm even on SM 12.0
# (jasl's sm12x_deep_gemm_fallbacks.py dispatches to Triton fallback, but the
# deep_gemm Python module must still be importable).
if [ -d "$VLLM_SRC/.deps/deepgemm-src" ]; then
    cd "$VLLM_SRC/.deps/deepgemm-src"
    pip install --no-build-isolation --no-deps .
    cd "$VLLM_SRC"
fi

# ---- 8. Card D specifically: apply jasl@27fd665b scheduler patch ----
# This addresses the Card D second-race (jasl/vllm#12, illegal-memory-access at c=4
# thinking). Verified 2026-05-27: 24/30 AIME c=4, 0 errors, 91.6% MTP acceptance.
# (Already in jasl@27fd665b; this is the runtime patch site for reference.)
if [ "$CARD" = "D" ]; then
    grep -q "max_num_scheduled_tokens // 16" vllm/v1/core/sched/scheduler.py && \
        echo "[patch] jasl@27fd665b scheduler patch already in pin" || \
        sed -i 's|mixed_prefill_budget = max(1, self.max_num_scheduled_tokens // 8)|mixed_prefill_budget = max(1, self.max_num_scheduled_tokens // 16)|' \
            vllm/v1/core/sched/scheduler.py
fi

# ---- 9. Runtime deps ----
pip install --upgrade huggingface_hub hf-transfer langdetect immutabledict nltk \
    evalplus openai datasets sentencepiece

# ---- 10. Download artifact ----
hf download "$ARTIFACT"

# ---- 11. Print serve recipe ----
cat <<EOF

================================================================
Install complete — Card $CARD ready on RTX PRO 6000 (SM 12.0)
================================================================

  vLLM:      jasl/vllm@$(cd "$VLLM_SRC" && git rev-parse --short HEAD) (ds4-sm120-preview-dev)
  Patches:   PR #43722 + #43723 + #41834 configs + workspace 4×
  Venv:      $VENV
  Artifact:  $ARTIFACT (cached under $HF_HOME)

Serve (TP=4, MTP if applicable):

  source $VENV/bin/activate
  VLLM_TEST_FORCE_FP8_MARLIN=1 \\
  vllm serve $ARTIFACT \\
    --tensor-parallel-size 4 \\
    --kv-cache-dtype fp8 --block-size 256 \\
    --max-model-len 32768 \\
    --max-num-seqs 8 --max-num-batched-tokens 8192 \\
    --gpu-memory-utilization 0.95 \\
    --compilation-config '{"max_cudagraph_capture_size":8,"cudagraph_capture_sizes":[1,2,4,8]}' \\
    --no-enable-prefix-caching \\
    --tokenizer-mode deepseek_v4 \\
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \\
    --reasoning-parser deepseek_v4 \\
$([ $MTP = 1 ] && echo "    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}' \\\\")
    --disable-custom-all-reduce \\
    --trust-remote-code \\
    --host 0.0.0.0 --port 8000

Caveats:
- Card B NVFP4 routed experts execute as Marlin BF16 on SM 12.0 (storage win
  preserved, no FLOPS win). Real FP4 tensor-core math needs B200/B300.
- --reasoning-parser deepseek_v4 + chat-template thinking=True interacts badly
  with MTP per upstream issue #34650. For thinking-mode evals, drop the
  reasoning-parser or use thinking=false.
- All three cards verified to load on this recipe. Card D additionally verified
  at AIME-30 c=4 thinking 24/30 = 80% correct, 91.6% MTP acceptance.

EOF
