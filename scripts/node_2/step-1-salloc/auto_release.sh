#!/bin/bash
# =============================================================================
# Auto-release GPU allocations when LongBench benchmark finishes.
# Monitors the results JSONL file; once it reaches the target count,
# kills vLLM servers and cancels both SLURM jobs.
#
# Usage: bash scripts/step-1-salloc/auto_release.sh [--target 503]
# Run this in a separate terminal. It polls every 30s.
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
RESULTS="/pscratch/sd/l/lingzhi/Projects/LongBench/results/Qwen3.5-27B.jsonl"
NODE_INFO="$WORKDIR/.node_info_multi_2"
TARGET=503
POLL_INTERVAL=30

# Parse args
if [[ "${1:-}" == "--target" ]] && [[ -n "${2:-}" ]]; then
    TARGET=$2
fi

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}=========================================="
echo "  Auto-Release Monitor"
echo "  Target: ${TARGET} completed questions"
echo "  Results: ${RESULTS}"
echo "  Poll interval: ${POLL_INTERVAL}s"
echo -e "==========================================${NC}"
echo ""

# Verify files exist
[ -f "$RESULTS" ] || { echo -e "${RED}Results file not found: $RESULTS${NC}"; exit 1; }
[ -f "$NODE_INFO" ] || { echo -e "${RED}Node info not found: $NODE_INFO${NC}"; exit 1; }

# Read job IDs
JOB_ID_1=$(grep SLURM_JOB_ID_1 "$NODE_INFO" | cut -d= -f2)
JOB_ID_2=$(grep SLURM_JOB_ID_2 "$NODE_INFO" | cut -d= -f2)
NODE_1=$(grep SLURM_NODE_1 "$NODE_INFO" | cut -d= -f2)
NODE_2=$(grep SLURM_NODE_2 "$NODE_INFO" | cut -d= -f2)

echo -e "  Job 1: ${JOB_ID_1} (${NODE_1})"
echo -e "  Job 2: ${JOB_ID_2} (${NODE_2})"
echo ""

while true; do
    DONE=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
    REMAIN=$((TARGET - DONE))
    NOW=$(date "+%H:%M:%S")

    if [ "$DONE" -ge "$TARGET" ]; then
        echo ""
        echo -e "${GREEN}[${NOW}] Target reached: ${DONE}/${TARGET}${NC}"
        echo ""

        # Kill vLLM servers
        echo -e "${YELLOW}Stopping vLLM servers...${NC}"
        srun --jobid="$JOB_ID_1" --overlap -N1 -n1 --gpus=4 \
            bash -c "pkill -9 -f 'vllm.entrypoints' 2>/dev/null; true" 2>/dev/null &
        srun --jobid="$JOB_ID_2" --overlap -N1 -n1 --gpus=4 \
            bash -c "pkill -9 -f 'vllm.entrypoints' 2>/dev/null; true" 2>/dev/null &
        wait 2>/dev/null
        sleep 3
        echo -e "${GREEN}  vLLM servers stopped.${NC}"

        # Cancel SLURM jobs
        echo -e "${YELLOW}Releasing GPU allocations...${NC}"
        scancel "$JOB_ID_1" 2>/dev/null && echo -e "${GREEN}  Job $JOB_ID_1 ($NODE_1) cancelled.${NC}" || echo -e "${YELLOW}  Job $JOB_ID_1 already gone.${NC}"
        scancel "$JOB_ID_2" 2>/dev/null && echo -e "${GREEN}  Job $JOB_ID_2 ($NODE_2) cancelled.${NC}" || echo -e "${YELLOW}  Job $JOB_ID_2 already gone.${NC}"

        echo ""
        echo -e "${GREEN}=========================================="
        echo "  All done! GPU nodes released."
        echo "  Results: ${DONE}/${TARGET}"
        echo -e "==========================================${NC}"
        exit 0
    fi

    printf "\r  [${NOW}] ${DONE}/${TARGET} done, ${REMAIN} remaining. Checking again in ${POLL_INTERVAL}s... "
    sleep "$POLL_INTERVAL"
done
