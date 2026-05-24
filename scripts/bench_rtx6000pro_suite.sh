#!/usr/bin/env bash
# Run the full benchmark suite against a running vllm serve instance.
# Drops outputs under benchmarks/rtx6000pro/<TS>/ — one dir per run.
#
# Usage:
#   bash scripts/bench_rtx6000pro_suite.sh <base_url> <tp> [skip_long]
#   skip_long=1 → skip MMLU + MMLU-Pro + AIME (the multi-hour items)

set -uo pipefail

if [[ -f "$HOME/venv-serve/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/venv-serve/bin/activate"
fi

BASE_URL="${1:?usage: $0 <base_url> <tp> [skip_long]}"
TP="${2:?usage: $0 <base_url> <tp> [skip_long]}"
SKIP_LONG="${3:-0}"
TS=$(date -u +%Y-%m-%dT%H%M%SZ)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/rtx6000pro/tp${TP}_${TS}"
MODEL_NAME="DSV4-NVFP4-FP8-MTP"
MODEL_PATH="/opt/dlami/nvme/weights/nvfp4-fp8-mtp"
mkdir -p "$OUT_DIR"
echo "[bench] writing to $OUT_DIR"

# --- chat-smoke quick (~1 min) ---
echo "[bench] chat-smoke quick"
python3 - "$BASE_URL" "$MODEL_NAME" >"$OUT_DIR/chat_smoke.log" 2>&1 <<'PYEOF'
import sys, json, urllib.request
base, model = sys.argv[1], sys.argv[2]
prompts = [
    "Write a Python function to reverse a string.",
    "What is 17 * 23?",
    "Explain quantum entanglement in 2 sentences.",
    "Translate 'Hello world' to French.",
]
ok = 0
for p in prompts:
    payload = {"model": model, "messages": [{"role":"user","content":p}],
               "max_tokens": 100, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", method="POST",
                                 headers={"Content-Type":"application/json"},
                                 data=json.dumps(payload).encode())
    try:
        r = json.load(urllib.request.urlopen(req, timeout=120))
        content = r["choices"][0]["message"].get("content","")
        if content and len(content) > 5:
            ok += 1
            print(f"OK [{len(content)}c]: {p!r} -> {content[:120]!r}")
        else:
            print(f"FAIL empty: {p!r}")
    except Exception as e:
        print(f"FAIL {e}: {p!r}")
print(f"\n[chat-smoke] {ok}/{len(prompts)} OK")
PYEOF
cat "$OUT_DIR/chat_smoke.log"

# --- MTP acceptance @ 100 (~3-5 min) ---
echo "[bench] MTP acceptance @ 100 prompts"
python3 - "$BASE_URL" "$MODEL_NAME" "$OUT_DIR/acceptance_100.json" 2>&1 | tee "$OUT_DIR/acceptance_100.log" <<'PYEOF'
import sys, json, time, random
from urllib.request import urlopen, Request
base, model, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
random.seed(42)
words = "the quick brown fox jumps over the lazy dog while pondering deep questions about life universe everything code math science art".split()
prompts = [" ".join(random.choices(words, k=48)) for _ in range(100)]

def get_metrics():
    out = {}
    try:
        text = urlopen(base + "/metrics", timeout=10).read().decode()
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#") or not line: continue
        if "spec_decode" in line or "draft_acceptance" in line or "num_accepted" in line or "num_draft" in line or "num_emitted" in line:
            parts = line.split()
            if len(parts) >= 2:
                try: out[parts[0]] = float(parts[-1])
                except: pass
    return out

m0 = get_metrics()
t0 = time.time()
for i, p in enumerate(prompts):
    payload = {"model": model, "prompt": p, "max_tokens": 200, "temperature": 0, "stream": False}
    req = Request(base + "/v1/completions", method="POST",
                  headers={"Content-Type":"application/json"},
                  data=json.dumps(payload).encode())
    try:
        urlopen(req, timeout=120).read()
    except Exception as e:
        print(f"prompt {i}: FAIL {e}")
    if (i+1) % 20 == 0:
        print(f"  {i+1}/100 done ({time.time()-t0:.0f}s)")

dt = time.time() - t0
m1 = get_metrics()
print(f"[mtp] total wall {dt:.0f}s")

result = {"n_prompts":100, "wall_s":dt, "before":m0, "after":m1}
for accepted_k in m1:
    if "num_accepted" in accepted_k:
        for draft_k in m1:
            if "num_draft" in draft_k or "num_proposed" in draft_k:
                d_acc = m1[accepted_k] - m0.get(accepted_k, 0)
                d_drf = m1[draft_k] - m0.get(draft_k, 0)
                if d_drf > 0:
                    result["acceptance_rate"] = d_acc / d_drf
                    result["num_accepted"] = d_acc
                    result["num_drafted"] = d_drf
                    print(f"[mtp] acceptance: {d_acc:.0f}/{d_drf:.0f} = {100*d_acc/d_drf:.2f}%")
                    break
        if "acceptance_rate" in result: break

with open(outpath, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"[mtp] wrote {outpath}")
PYEOF

# --- Throughput TPOT (vllm bench serve, MTP-on, bs=1/4/16) ---
echo "[bench] throughput TPOT (MTP-on, bs=1/4/16) via vllm bench serve"
for BS in 1 4 16; do
    NREQ=$((BS * 4))
    vllm bench serve \
        --base-url "$BASE_URL" \
        --model "$MODEL_NAME" \
        --tokenizer "$MODEL_PATH" \
        --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 256 \
        --num-prompts $NREQ --max-concurrency $BS \
        --save-result --result-dir "$OUT_DIR" \
        --result-filename "bench_mtp_bs${BS}.json" 2>&1 | tee "$OUT_DIR/bench_mtp_bs${BS}.log" || \
        echo "[warn] vllm bench bs=$BS failed"
done

# --- GSM8K 50-prompt smoke (8-shot) ---
echo "[bench] GSM8K 50-prompt smoke (8-shot)"
lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False" \
    --num_fewshot 8 \
    --limit 50 \
    --output_path "$OUT_DIR/gsm8k50.json" \
    --log_samples 2>&1 | tee "$OUT_DIR/gsm8k50.log" || echo "[warn] GSM8K50 failed"

if [[ "$SKIP_LONG" != "1" ]]; then
    # --- HumanEval pass@1 ---
    echo "[bench] HumanEval pass@1"
    lm_eval --model local-completions \
        --tasks humaneval_instruct \
        --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False" \
        --confirm_run_unsafe_code \
        --output_path "$OUT_DIR/humaneval.json" 2>&1 | tee "$OUT_DIR/humaneval.log" || echo "[warn] HumanEval failed"

    # --- GSM8K full (8-shot, ~30 min) ---
    echo "[bench] GSM8K full 8-shot"
    lm_eval --model local-completions \
        --tasks gsm8k \
        --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False" \
        --num_fewshot 8 \
        --output_path "$OUT_DIR/gsm8k.json" 2>&1 | tee "$OUT_DIR/gsm8k.log" || echo "[warn] GSM8K full failed"
fi

echo "[bench] DONE. Output in $OUT_DIR"
ls -la "$OUT_DIR"
