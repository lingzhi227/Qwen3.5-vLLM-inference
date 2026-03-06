#!/bin/bash
# =============================================================================
# Stress test across 4 vLLM servers (4x A100 GPUs)
# Runs parallel stress tests, one per GPU, then aggregates results
#
# Usage:
#   bash scripts/test_stress_multi.sh              # default: 50 agents per GPU (200 total)
#   bash scripts/test_stress_multi.sh 30            # 30 per GPU (120 total)
#   bash scripts/test_stress_multi.sh 10 30 50 80   # ramp-up per GPU
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info_multi"
MODEL="/pscratch/sd/l/lingzhi/models/Qwen3.5-27B"

# Get server info
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi

if [ -z "${API_URLS:-}" ]; then
    echo "ERROR: No multi-GPU server info found. Run step2_serve_multi.sh first."
    exit 1
fi

# Parse API URLs
IFS=',' read -ra URLS <<< "$API_URLS"
NUM_GPUS=${#URLS[@]}

echo "============================================"
echo "  Multi-GPU Stress Test"
echo "  GPUs: $NUM_GPUS"
for i in "${!URLS[@]}"; do
    echo "  GPU $i: ${URLS[$i]}"
done
echo "============================================"
echo ""

# Health check all servers
ALIVE=0
for i in "${!URLS[@]}"; do
    url="${URLS[$i]}"
    if curl -s --connect-timeout 5 "${url}/models" > /dev/null 2>&1; then
        echo "[OK] GPU $i responding"
        ALIVE=$((ALIVE+1))
    else
        echo "[FAIL] GPU $i NOT responding: $url"
    fi
done

if [ $ALIVE -eq 0 ]; then
    echo "ERROR: No servers responding"
    exit 1
fi
echo ""
echo "$ALIVE / $NUM_GPUS servers alive"
echo ""

# Install deps
python3 -m pip show openai &>/dev/null || python3 -m pip install --user openai
python3 -m pip show cwltool &>/dev/null || python3 -m pip install --user cwltool

CWLTOOL_BIN=$(python3 -c "import shutil; print(shutil.which('cwltool') or '')")
if [ -z "$CWLTOOL_BIN" ]; then
    CWLTOOL_BIN="$HOME/.local/bin/cwltool"
fi

# Concurrency levels
if [ $# -gt 0 ]; then
    LEVELS="$@"
else
    LEVELS="50"
fi

echo "Concurrency per GPU: $LEVELS"
echo "Total agents per level: $(echo $LEVELS | awk '{for(i=1;i<=NF;i++) printf "%d ", $i*'$ALIVE'}')"
echo ""

# Run stress test on each GPU in parallel
STRESS_SCRIPT="$WORKDIR/tests/coding_cwl/cwl_stress_test.py"
PIDS=()
LOG_DIR="$WORKDIR/stress_multi_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for i in "${!URLS[@]}"; do
    url="${URLS[$i]}"
    if ! curl -s --connect-timeout 3 "${url}/models" > /dev/null 2>&1; then
        echo "[SKIP] GPU $i not responding"
        continue
    fi

    log="$LOG_DIR/gpu${i}_${TIMESTAMP}.log"
    echo "Starting stress test on GPU $i → $log"

    VLLM_API_URL="$url" \
    VLLM_MODEL="$MODEL" \
    CWLTOOL_BIN="$CWLTOOL_BIN" \
    STRESS_WORKSPACE_ID="gpu${i}" \
    python3 "$STRESS_SCRIPT" -n $LEVELS \
        > "$log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} stress tests launched. Waiting ..."
echo ""

# Wait for all to complete
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        echo "[DONE] GPU $i stress test completed (PID $pid)"
    else
        echo "[FAIL] GPU $i stress test failed (PID $pid)"
        FAILED=$((FAILED+1))
    fi
done

echo ""
echo "============================================"
echo "  All tests complete ($FAILED failures)"
echo "============================================"
echo ""

# Print summary from each GPU log
for i in "${!URLS[@]}"; do
    log="$LOG_DIR/gpu${i}_${TIMESTAMP}.log"
    if [ -f "$log" ]; then
        echo "── GPU $i ──────────────────────────────────"
        # Extract the ramp-up summary or last summary block
        grep -A 20 "RAMP-UP SUMMARY\|Summary (" "$log" | tail -30
        echo ""
    fi
done

# Aggregate throughput across GPUs
echo "── AGGREGATE ──────────────────────────────"
for level in $LEVELS; do
    total_gen_tps=0
    total_success=0
    total_agents=0
    for i in "${!URLS[@]}"; do
        log="$LOG_DIR/gpu${i}_${TIMESTAMP}.log"
        if [ -f "$log" ]; then
            # Extract gen tok/s for this level
            tps=$(grep -A 5 "Throughput:" "$log" | grep "gen tok/s" | tail -1 | grep -oP '[\d.]+(?= gen tok/s)' || echo "0")
            # Extract success count
            success_line=$(grep "Success:" "$log" | tail -1)
            if [ -n "$success_line" ]; then
                s=$(echo "$success_line" | grep -oP '\d+(?=/)')
                n=$(echo "$success_line" | grep -oP '(?<=/)\d+')
                total_success=$((total_success + s))
                total_agents=$((total_agents + n))
            fi
            total_gen_tps=$(echo "$total_gen_tps + $tps" | bc)
        fi
    done
    echo "  ${ALIVE}x GPU × $level agents = $total_agents total"
    echo "  Success: $total_success / $total_agents"
    echo "  Combined gen throughput: ${total_gen_tps} tok/s"
    echo ""
done
