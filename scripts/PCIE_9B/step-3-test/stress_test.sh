#!/bin/bash
# =============================================================================
# Stress test: measure throughput, latency, and max batch on A100-PCIE-40GB
# Usage:
#   bash scripts/PCIE/stress_test.sh             # default ramp-up
#   bash scripts/PCIE/stress_test.sh 1 2 4 8     # custom concurrency levels
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info_pcie"

# Get API URL
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi
API_URL="${API_URL:-http://localhost:8000/v1}"

echo "Stress testing vLLM at: $API_URL"

if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo "ERROR: Server not responding at $API_URL"
    exit 1
fi

# Concurrency levels
if [ $# -gt 0 ]; then
    LEVELS="$*"
else
    LEVELS="1 2 4 8 16"
fi

echo "Concurrency levels: $LEVELS"
echo ""

export API_URL
export LEVELS

python3 << 'PYEOF'
import os, sys, time, json, threading, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

API_URL = os.environ["API_URL"]
LEVELS = [int(x) for x in os.environ["LEVELS"].split()]

client = OpenAI(base_url=API_URL, api_key="empty")
model = client.models.list().data[0].id
print(f"Model: {model}")
print()

# ---------------------------------------------------------------------------
# Test prompts of varying length
# ---------------------------------------------------------------------------
SHORT_PROMPT = "Write a haiku about coding."
MEDIUM_PROMPT = "Explain the difference between RNA-seq and microarray gene expression analysis. Cover key technical differences, advantages, and disadvantages of each approach."
LONG_PROMPT = "You are a bioinformatics expert. Describe in detail the complete pipeline for performing differential gene expression analysis using DESeq2, starting from raw FASTQ files. Include quality control, alignment, counting, normalization, statistical testing, and downstream analysis like GO enrichment. Provide specific R code examples for each step."

OUTPUT_LENS = {
    "short": (SHORT_PROMPT, 64),
    "medium": (MEDIUM_PROMPT, 256),
    "long": (LONG_PROMPT, 1024),
}

# ---------------------------------------------------------------------------
# Single request benchmark
# ---------------------------------------------------------------------------
def bench_single(prompt, max_tokens, enable_thinking=True):
    """Run a single request and return metrics."""
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
            extra_body={"enable_thinking": enable_thinking},
        )
        elapsed = time.time() - start
        usage = resp.usage
        return {
            "ok": True,
            "elapsed": elapsed,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": (usage.prompt_tokens + usage.completion_tokens) if usage else 0,
            "tok_per_sec": (usage.completion_tokens / elapsed) if usage and elapsed > 0 else 0,
        }
    except Exception as e:
        return {"ok": False, "elapsed": time.time() - start, "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Phase 1: Single-request latency & speed for different output lengths
# ---------------------------------------------------------------------------
print("=" * 70)
print("  Phase 1: Single-request performance (no concurrency)")
print("=" * 70)
print(f"{'Task':<10} {'Prompt Tok':>10} {'Gen Tok':>10} {'Time (s)':>10} {'Tok/s':>10}")
print("-" * 55)

for name, (prompt, max_tok) in OUTPUT_LENS.items():
    r = bench_single(prompt, max_tok)
    if r["ok"]:
        print(f"{name:<10} {r['prompt_tokens']:>10} {r['completion_tokens']:>10} {r['elapsed']:>10.2f} {r['tok_per_sec']:>10.1f}")
    else:
        print(f"{name:<10} {'ERROR':>10} {r['error'][:30]}")
print()


# ---------------------------------------------------------------------------
# Phase 2: Concurrent throughput test
# ---------------------------------------------------------------------------
print("=" * 70)
print("  Phase 2: Concurrent throughput (medium prompt, 256 max tokens)")
print("=" * 70)
print(f"{'Conc':>5} {'Requests':>10} {'OK':>5} {'Fail':>5} {'Total(s)':>10} {'Avg Lat':>10} {'P50 Lat':>10} {'P95 Lat':>10} {'Total Tok/s':>12}")
print("-" * 85)

for conc in LEVELS:
    n_requests = max(conc * 2, 4)  # at least 4 requests, 2x concurrency
    results = []

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futures = [pool.submit(bench_single, MEDIUM_PROMPT, 256) for _ in range(n_requests)]
        for f in as_completed(futures):
            results.append(f.result())
    total_time = time.time() - start_all

    ok_results = [r for r in results if r["ok"]]
    fail_count = len(results) - len(ok_results)

    if ok_results:
        lats = [r["elapsed"] for r in ok_results]
        total_gen_tokens = sum(r["completion_tokens"] for r in ok_results)
        avg_lat = statistics.mean(lats)
        p50 = statistics.median(lats)
        p95 = sorted(lats)[int(0.95 * len(lats))] if len(lats) > 1 else lats[0]
        total_tps = total_gen_tokens / total_time

        print(f"{conc:>5} {n_requests:>10} {len(ok_results):>5} {fail_count:>5} {total_time:>10.2f} {avg_lat:>10.2f} {p50:>10.2f} {p95:>10.2f} {total_tps:>12.1f}")
    else:
        print(f"{conc:>5} {n_requests:>10} {0:>5} {fail_count:>5} {total_time:>10.2f} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>12}")

    if fail_count > len(results) // 2:
        print(f"  [STOP] Too many failures at concurrency {conc}")
        break
print()


# ---------------------------------------------------------------------------
# Phase 3: Long context test
# ---------------------------------------------------------------------------
print("=" * 70)
print("  Phase 3: Long output generation (1024 tokens, concurrency 1 vs 4)")
print("=" * 70)

for conc in [1, 4]:
    results = []
    start_all = time.time()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futures = [pool.submit(bench_single, LONG_PROMPT, 1024) for _ in range(conc)]
        for f in as_completed(futures):
            results.append(f.result())
    total_time = time.time() - start_all

    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        total_gen = sum(r["completion_tokens"] for r in ok_results)
        avg_speed = statistics.mean([r["tok_per_sec"] for r in ok_results])
        total_tps = total_gen / total_time
        print(f"  Concurrency {conc}: {total_gen} tokens in {total_time:.1f}s")
        print(f"    Per-request: {avg_speed:.1f} tok/s | Aggregate: {total_tps:.1f} tok/s")
    else:
        print(f"  Concurrency {conc}: FAILED")
print()


# ---------------------------------------------------------------------------
# Phase 4: Max batch probe (keep increasing until failure)
# ---------------------------------------------------------------------------
print("=" * 70)
print("  Phase 4: Max batch probe (short prompt, 64 tokens)")
print("=" * 70)

max_ok_batch = 0
for batch in [1, 4, 8, 16, 32, 64]:
    results = []
    start_all = time.time()
    with ThreadPoolExecutor(max_workers=batch) as pool:
        futures = [pool.submit(bench_single, SHORT_PROMPT, 64, False) for _ in range(batch)]
        for f in as_completed(futures):
            results.append(f.result())
    total_time = time.time() - start_all

    ok_count = sum(1 for r in results if r["ok"])
    fail_count = batch - ok_count
    total_gen = sum(r.get("completion_tokens", 0) for r in results if r["ok"])
    tps = total_gen / total_time if total_time > 0 else 0

    status = "OK" if fail_count == 0 else f"FAIL({fail_count})"
    print(f"  Batch {batch:>3}: {status:<10} {total_gen:>6} tokens in {total_time:>6.1f}s = {tps:>8.1f} tok/s")

    if ok_count > 0:
        max_ok_batch = batch
    if fail_count > batch // 2:
        break

print(f"\n  Max reliable batch size: {max_ok_batch}")
print()

print("=" * 70)
print("  Stress test complete!")
print("=" * 70)
PYEOF
