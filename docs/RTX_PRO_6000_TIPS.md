# RTX PRO 6000 deployment tips

Everything we learned getting `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` running end-to-end on RTX PRO 6000 Blackwell Server Edition (SM 12.0) in a fresh Docker container. The recipe is real — [Docker quickstart in README](../README.md) — this doc is the *why behind each setting and the gotchas you'll hit if you deviate*.

## Hardware reality check first

| Aspect | Reality on RTX PRO 6000 |
|---|---|
| On-disk + HBM footprint | NVFP4 packed weights ✓ (172 GB vs 600 GB BF16) |
| MoE compute | **Marlin BF16** (Marlin dequantizes FP4→BF16 inside kernel) — **not** native FP4 tensor cores |
| FP8 attention compute | ✓ native FP8 tensor cores via `TritonFp8BlockScaledMMKernel` |
| MTP spec-decode | ✓ functional, ~1.7-2× decode speedup |

So you get **memory savings**, not **FLOPS savings**, from NVFP4 on RTX PRO 6000 today. Native FP4 FLOPS needs B200/B300 or wait for CUTLASS sm_120 fix (open issues #3096 + vLLM #41738, #43333, #43341, #43687).

Verify the chosen MoE kernel in your serve log:

```
INFO [nvfp4.py:231] Using 'MARLIN' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM']
```

`MARLIN` = today's path. `FLASHINFER_TRTLLM` = future native-FP4 path.

## Three serve profiles, pick by use case

All three use the same image (`canada-quant/dsv4-rtx6000pro:v3`) and same patches. Profile differences are runtime config only.

| Profile | Script | max_model_len | max_num_seqs | Use case |
|---|---|---|---|---|
| Standard | `scripts/serve_rtx6000pro_tp2.sh` | 8192 | 2 | Chat, RAG, agentic, multi-turn (each turn ≤8K) |
| **Long reasoning** | `scripts/serve_rtx6000pro_tp2_32k.sh` | 32768 | 1 | **AIME-style benchmarks, deep thinking-mode** |
| Max context | `scripts/serve_rtx6000pro_tp2_longctx.sh` | 131072 | 1 | Single-user long-document RAG, ≤128K input |

The 4-GPU variant (`serve_rtx6000pro_tp4.sh`) exists but isn't the focus here.

## The required env vars (don't skip)

```bash
VLLM_TEST_FORCE_FP8_MARLIN=1     # routes NVFP4 MoE to Marlin (only path that works on SM 12.0)
VLLM_USE_LAYERNAME=0             # disables jasl's LayerName opaque type
                                 # — bypasses an Inductor lowering crash on
                                 # vllm.moe_forward_shared.default (FakeScriptObject
                                 # .stride access). Without this you need
                                 # --enforce-eager which kills CUDA graph perf.
HF_HUB_ENABLE_HF_TRANSFER=1      # speeds artifact download
```

## CUDA graph capture sizes: must be multiple of (num_speculative_tokens + 1)

If you set `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (the recommended MTP config), then `cudagraph_capture_sizes` entries must each be multiples of **2** (= 1+1). Setting `[1]` produces:

```
No valid cudagraph sizes after rounding to multiple of 2 (num_speculative_tokens + 1)
```

Use `[2]` for single-stream serve, `[1,2]` for max_num_seqs=2, etc. The `1` value works only when speculative is off.

## Memory budgeting at TP=2

Math per GPU (96 GB each):
- Card B weights at TP=2: ~86 GB/GPU
- MTP shared block: ~6.5 GB/GPU
- Used by weights+MTP: **~92.5 GB/GPU**
- Free for KV + cudagraph + activations: **~3.5 GB/GPU**

DSv4 compressed-MLA + FP8 KV cache costs ~16 KB/token/GPU at TP=2 (after the TP split). So:

| max_model_len | KV pool/GPU | Fits in 3.5 GB free? |
|---|---|---|
| 8K × 2 seqs | 0.25 GB | ✓ easy |
| 16K × 2 seqs | 0.5 GB | ✓ |
| 32K × 1 seq | 0.5 GB | ✓ |
| **128K × 1 seq** | **2 GB** | ✓ leaves ~1 GB for cudagraph |
| 256K × 1 seq | 4 GB | ✗ over budget |

If you want >128K context, drop MTP (frees ~6.5 GB/GPU) and set `max_model_len` up to ~256K.

## Calling thinking mode correctly (NOT broken — easy to misread)

vLLM 0.21+ renamed the OpenAI-spec field `reasoning_content` to **`reasoning`**. Older client code reading `reasoning_content` will see `None` even when the server populated the reasoning field. This caused us a multi-hour false-positive "MTP × thinking parser bug" investigation — the parser was fine, our test was wrong.

Correct request:

```json
{
  "model": "DSV4-NVFP4-FP8-MTP",
  "messages": [{"role": "user", "content": "Find the smallest n ..."}],
  "max_tokens": 4000,
  "temperature": 0,
  "chat_template_kwargs": {"thinking": true},
  "reasoning_effort": "high",
  "include_reasoning": true
}
```

Read the response from `choices[0].message.reasoning` (not `reasoning_content`).

Verified levels (all return correct answers on `n^2 + n + 41` composite probe → n=40):

| `reasoning_effort` | reasoning chars | content chars | tokens used |
|---|---|---|---|
| `off` (thinking=false) | 0 | 1840 | 815 |
| `high` | 3776 | 359 | 1499 |
| `max` | 6288 | 397 | 2824 |

## Thinking + AIME: budget `max_tokens=32000` or you'll truncate

AIME-2024 thinking-mode chains average ~5-15K tokens; some hit 30K+. With `max_tokens=8000` we got **11/30 length-truncations** and the raw pass@1 dropped to 17/30 (vs baseline 24/30). With `max_tokens=32000` we got **6/30 length-truncations** and 24/30 — exact baseline match.

If you bench AIME on RTX PRO 6000, use `serve_rtx6000pro_tp2_32k.sh` and pass `max_tokens=32000` to the bench client.

## bs=1 is the sweet spot — bs=2+ scales poorly

Measured TP=2 throughput on `random 256-in/256-out`:

| Concurrency | Avg out tok/s | Per-stream | MTP accept |
|---|---|---|---|
| bs=1 | 45 / 73 peak | 45-73 | 71% |
| bs=2 | 21 | 10.5 | 74% |

Per-stream throughput drops 4× at bs=2. Diagnosis: TP-allreduce comm-bound over PCIe (no NVLink) + Marlin BF16 MoE saturating the BF16 tensor cores. For multi-user serving on this hardware, **run two replicas at bs=1** instead of one replica at bs=2. For real concurrency, use TP=4 on 4× cards or B200/B300.

## If you must build the Docker image from source

The build will compile 366 CUDA objects (~30-45 min on 12 vCPUs) the first time. Cache reuses across rebuilds. Image is ~14 GB.

```bash
git clone https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp.git
cd dsv4-flash-nvfp4-fp8-mtp
docker build \
  -f docker/Dockerfile.rtx6000pro \
  -t canada-quant/dsv4-rtx6000pro:v3 \
  .
```

NGC base image is pinned to `nvcr.io/nvidia/pytorch@sha256:192d749b4d773610ec9e01c0443a9df545d196c412b7b8fd33bfa3da362a49e7` (specific digest, not the mutable `26.04-py3` tag). vLLM SHA `27fd665b...` (full 40-char), canada-quant cherry-pick `5a49d8803 + 5acabf315` (full SHAs). All jasl runtime kernels pinned (humming-kernels==0.1.2, flashinfer-python==0.6.11.post3, nvidia-cutlass-dsl==4.5.0, etc.). A force-push on upstream branches **cannot** break this image.

## Disk layout matters on a fresh AWS DLAMI

Root partition on a fresh `g7e.24xlarge` is 248 GB with only ~11 GB free after the DLAMI install. Move BOTH Docker and containerd off `/`:

```bash
sudo systemctl stop docker docker.socket containerd
sudo mkdir -p /opt/dlami/nvme/docker /opt/dlami/nvme/containerd
sudo rsync -a /var/lib/containerd/ /opt/dlami/nvme/containerd/
sudo rm -rf /var/lib/containerd
sudo ln -s /opt/dlami/nvme/containerd /var/lib/containerd

# /etc/docker/daemon.json:
# {
#   "data-root": "/opt/dlami/nvme/docker",
#   "default-runtime": "nvidia",
#   "mtu": 9001,
#   "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "args": []}}
# }

sudo systemctl start containerd docker
```

Without this you'll hit `no space left on device` during the NGC base image pull (~5 GB).

## The 13 patches the Docker bakes (FYI)

| # | Source | What it fixes |
|---|---|---|
| 1 | PR #43722 | MarlinFP8 refuses block-FP8 layers |
| 2 | PR #43723 | DSv4 wo_a scale getattr fallback |
| 3 | PR #41834 | Tuned Triton block-FP8 configs for SM 12.0 |
| 4a | repo-local | Marlin MoE workspace 4× oversize (defensive) |
| 4b | repo-local | `has_tuple_return = False` (NGC torch lacks the kwarg) |
| 4c | repo-local | `cute.arch.fmin` shim (cutlass-dsl 4.5.0 lacks it) |
| 4d-e | canada-quant/vllm cherry-pick | BF16 MTP detect + wo_a BF16 ref path (issue #43304) |
| 5 | PR #40923 | MARLIN_MOE_ARCHS extended to sm_120a / sm_121a |
| 6 | dep | python3-yaml apt-purge before pip |
| 7 | dep | torch pin via PIP_CONSTRAINT (prevents PyPI overriding NGC fork) |
| 8 | dep | packaging>=24.2 via PIP_CONSTRAINT= override of NGC's 23.2 pin |
| 9 | dep | setuptools_rust + setuptools_scm (NGC base lacks them) |

Detailed layer-by-layer archaeology in [`docs/findings/cardb_docker_layers_2026_05_28.md`](findings/cardb_docker_layers_2026_05_28.md).

## What we benched + verified end-to-end (2× RTX PRO 6000, 2026-05-28)

| Test | Result |
|---|---|
| AIME-2024 c=1 thinking max_tokens=32K | 24/30 (matches baseline exactly), MTP 91.05% |
| Throughput bs=1 random 256/256 | 45/73 tok/s, MTP 71% |
| Tool calling (`get_weather`) | ✓ structured emit |
| Thinking mode all 3 effort levels | ✓ all return correct answers |
| 128K context single-user | ✓ verified, 93 GB/GPU |
| Smoke (math/code/word problem) | ✓ all correct |

Raw JSON in `benchmarks/rtxpro6000/tp2_*_2026_05_28.*`.

## When you hit a problem

1. Check `nvidia-smi` for GPU 0/1 memory usage. 86 GB = weights only. >92 GB and rising = KV cache filling. 95+ GB constant = engine paused (CUDA OOM imminent).
2. `docker logs cardb-tp2 | grep -E "ERROR|Worker failed|RuntimeError"` — most failures we hit are dep issues at startup, not runtime.
3. The container needs `--shm-size=16g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864` — without these vLLM workers hang on multiproc executor init.
4. If `/v1/models` doesn't return after 15 min, weights are still loading or CUDA graph capture is in progress. The `shm_broadcast` "No available shared memory broadcast block found in 60 seconds" messages are normal during graph capture.

## Cross-references

- Repo: <https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp>
- HF model card: <https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP>
- v3 install script: [`scripts/install_rtx6000pro_v3.sh`](../scripts/install_rtx6000pro_v3.sh) (host install, no Docker)
- Docker recipe: [`docker/Dockerfile.rtx6000pro`](../docker/Dockerfile.rtx6000pro)
- Layer-by-layer archaeology: [`docs/findings/cardb_docker_layers_2026_05_28.md`](findings/cardb_docker_layers_2026_05_28.md)
- TP=2 benchmark summary: [`benchmarks/rtxpro6000/tp2_summary_2026_05_28.md`](../benchmarks/rtxpro6000/tp2_summary_2026_05_28.md)
