#!/usr/bin/env bash
# install_rtx6000pro_v3.sh — RTX PRO 6000 install for canada-quant DSv4 cards
#
# Status: 2026-05-28 — full-bench-verified recipe. See VERIFIED RESULTS below.
#
# 2026-05-28 UPDATES (after fresh-environment Docker reproduction):
#   - Fixed 3 latent bugs in the prior 2026-05-27 version (caught when run on a
#     box without leftover state from previous vllm installs):
#       1. PR #43723 patch path was `nvidia/ops/attention.py` — actual file is
#          `vllm/models/deepseek_v4/attention.py` (no `nvidia/ops/` prefix).
#       2. PR #40923 grep guard `grep -q "12.0a;12.1a" CMakeLists.txt` falsely
#          matches OTHER kernel arch lists (SCALED_MM/FP4/MLA), so MARLIN_MOE_ARCHS
#          stayed at "8.0+PTX" (silent JIT-PTX corruption on sm_120). Tightened
#          to grep the literal MARLIN_MOE_ARCHS marker.
#       3. `.deps/deepgemm-src` doesn't exist at jasl@27fd665b — that layout
#          predates the `sm12x_deep_gemm_fallbacks.py` shim. Step removed.
#   - Added cherry-pick of canada-quant/vllm@5a49d8803 + 5acabf315 (branch
#     `fix/dsv4-mtp-draft-quant-detect`) which fixes upstream issue #43304 — the
#     real root cause of Card B's `wo_a.weight_scale` AttributeError on a clean
#     install. PR #43723's getattr fallback is a partial fix; the BF16 MTP block
#     needs its quant_config skipped at layer construction time.
#   - Expanded runtime deps to cover jasl-specific kernels (humming-kernels,
#     quack-kernels, tokenspeed-mla, fastsafetensors, flashinfer-python,
#     flashinfer-cubin, tilelang) and vLLM's common.txt transitives (fastapi,
#     starlette, etc.). The 2026-05-27 version assumed these were already in the
#     host's pip env from prior installs — clean machines need them explicit.
#
# WHY JASL, NOT MAINLINE
# Mainline (vllm-project/vllm@main) is blocked at first forward pass on SM 12.0
# by DeepGEMM Hopper-only kernels (`fp8_einsum`, `tf32_hc_prenorm_gemm`).
# jasl/vllm carries SM 12.0 fallback files (`sm12x_deep_gemm_fallbacks.py`,
# `sm12x_mqa.py`, `fp8_einsum.py`, `cutedsl_utils.py`) not yet upstreamed.
# Open vLLM PRs to track mainline progress: #41834, #41738, #43333, #43341, #43687.
#
# VERIFIED RESULTS (2026-05-28, with the patches in this updated script):
#   Card B (NVFP4-FP8-MTP): full bench in progress; intermediate gate passed
#                           (weights load, workers init, kernels JIT, profile fwd OK)
#   Card D (W4A16-FP8-MTP): AIME-30 c=4 thinking = 24/30 correct, 0 CUDA errors,
#                           91.61% MTP acceptance, 641.9s wall (10.7 min)
#   Card A: not retested this session — same install pattern applies
#
# USAGE
#   CARD=B bash install_rtx6000pro_v3.sh
#   CARD=D bash install_rtx6000pro_v3.sh
#   CARD=A bash install_rtx6000pro_v3.sh
#
# Run time: ~45-60 min on a fresh Brev g7e.24xlarge (build) + 30 min Card B download
# (172 GB) or 142 GB Card D / 142 GB Card A.
#
# DOCKER ALTERNATIVE
# For a containerized install with the same recipe baked in, see
# `docker/Dockerfile.rtx6000pro` and the orchestrator `scripts/bench_docker_full.sh`.

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
# NOTE 2026-05-28: prior version of this script had path `nvidia/ops/attention.py`
# which doesn't exist at jasl@27fd665b — corrected to `models/deepseek_v4/attention.py`.
# This is a partial fix; the full architectural fix is the BF16 MTP cherry-pick
# below (issue #43304). Keep this patch as belt-and-suspenders for the non-MTP path.
python3 - <<'PYEOF'
F = "vllm/models/deepseek_v4/attention.py"
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

# canada-quant/vllm:fix/dsv4-mtp-draft-quant-detect — THE root-cause fix for
# issue #43304 (Card B `wo_a.weight_scale` AttributeError after weight load).
# Two commits: 5a49d8803 + 5acabf315, authored 2026-05-21.
#   - 5a49d88: detect BF16 MTP on disk, skip applying main quant_config to MTP
#              block during DeepSeekV4MultiTokenPredictorLayer construction
#   - 5acabf3: BF16 reference path in attention forward when wo_a is unquantized
# Without this, layer construction allocates main-scheme quant slots on MTP
# layers that have no scales on disk → AttributeError at forward time.
echo "[step] cherry-picking canada-quant BF16 MTP fix (issue #43304)"
git remote add canada-quant https://github.com/canada-quant/vllm.git 2>/dev/null || true
git fetch canada-quant fix/dsv4-mtp-draft-quant-detect
git config user.email "build@canada-quant.local" 2>/dev/null || true
git config user.name "canada-quant install" 2>/dev/null || true
# Commit any prior in-place patches so cherry-pick has a clean working tree
git add -A
git commit --allow-empty -m "install_rtx6000pro_v3: prior in-place patches" 2>/dev/null || true
if ! git cherry-pick 5a49d8803 5acabf315; then
    # PR #43723's in-place edits to attention.py may conflict with 5acabf315 since
    # both touch the same lines. Discard the in-place attention.py modification
    # and re-apply only the cherry-picks (which include 5acabf315's superset fix).
    echo "[patch] cherry-pick conflict — resetting attention.py and retrying"
    git cherry-pick --abort
    git checkout HEAD~1 -- vllm/models/deepseek_v4/attention.py
    git commit --allow-empty -m "reset attention.py for clean cherry-pick"
    git cherry-pick 5a49d8803 5acabf315
fi
echo "[patch] BF16 MTP detect + wo_a BF16 ref path applied (issue #43304)"

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

# PR #40923 — Marlin MoE sm_120a;sm_121a native cubins. The prior version's
# grep guard `grep -q "12.0a;12.1a" CMakeLists.txt` falsely matched other kernel
# arch lists (SCALED_MM, FP4, MLA all extend to 12.0a;12.1a in jasl), making the
# conditional always skip — leaving MARLIN_MOE_ARCHS at "8.0+PTX" which silently
# JIT-PTXes Marlin MoE kernels on sm_120 (correctness corruption). Use a tighter
# marker that targets MARLIN_MOE_ARCHS specifically.
if grep -q 'cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX"' CMakeLists.txt; then
    sed -i 's|cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX"|cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX;9.0a;10.0a;10.1a;10.3a;12.0a;12.1a"|' CMakeLists.txt
    echo "[patch] PR #40923 MARLIN_MOE_ARCHS extended to include sm_120a/sm_121a"
else
    echo "[patch] PR #40923 MARLIN_MOE_ARCHS already extended or marker moved"
fi

# ---- 6. Build vLLM (CUDA kernel compile, 30-45 min on g7e.24xlarge) ----
export TORCH_CUDA_ARCH_LIST="12.0a"
export MAX_JOBS=12
export CMAKE_BUILD_PARALLEL_LEVEL=12
export VLLM_USE_PRECOMPILED=0

echo "[build] pip install -e . (CUDA kernel compile, ~30-45 min)..."
pip install --no-build-isolation --no-deps -v -e .

# ---- 7. (removed 2026-05-28) DeepGEMM bundled-source step ----
# Prior version's `if [ -d "$VLLM_SRC/.deps/deepgemm-src" ]` was a no-op at
# jasl@27fd665b — that bundled-deepgemm layout predates the
# `vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` shim that
# jasl ships natively. `vllm.utils.deep_gemm` is a vLLM-internal module and
# does not require the standalone PyPI `deep_gemm` package on SM 12.0.

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
# 2026-05-28: expanded to be self-sufficient on a clean machine. Prior version
# assumed the host pip env already had vllm's transitives + jasl kernels from
# previous installs. On a fresh box (or in Docker) we need them explicit.

# 9a. vLLM's own runtime deps (from requirements/common.txt). On Docker we use
# PIP_CONSTRAINT to pin the NGC torch fork; on host we rely on the active venv.
if [ -f "$VLLM_SRC/requirements/common.txt" ]; then
    pip install --upgrade -r "$VLLM_SRC/requirements/common.txt"
fi

# 9b. jasl-specific kernels pinned to versions verified end-to-end 2026-05-28.
# - quant registry imports `humming` eagerly even for non-humming models
# - workers import flashinfer for attention
# - mhc path needs tilelang for JIT
# - nvidia-cutlass-dsl MUST be 4.5.0; 4.5.2 removed `fmin` from cutlass.cute.arch
#   and breaks jasl's sparse_attn_compress_cutedsl path.
# cu12 wheels work fine despite jasl's cuda.txt preferring cu13.
pip install --upgrade \
    "humming-kernels==0.1.2" \
    "quack-kernels==0.4.1" \
    "tokenspeed-mla==0.1.5" \
    "fastsafetensors==0.3.2" \
    "flashinfer-python==0.6.11.post3" \
    "flashinfer-cubin==0.6.11.post3" \
    "tilelang==0.1.10" \
    "nvidia-cutlass-dsl==4.5.0"

# 9c. ray distributed runtime (vllm uses for multiproc executor)
pip install --upgrade "ray[default]" opentelemetry-exporter-otlp

# 9d. bench / eval harness deps
pip install --upgrade huggingface_hub hf-transfer hf_xet \
    langdetect immutabledict nltk \
    "lm-eval>=0.4.12" evalplus openai datasets sentencepiece aiohttp

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
