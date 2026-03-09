#!/bin/bash
# =============================================================================
# Stop vLLM servers interactively
# Lists all running vLLM server processes and lets you choose which to stop.
# Usage: bash scripts/step-2-vLLM/stop_vllm.sh
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
BOLD='\033[1m'

# ---------------------------------------------------------------------------
# Discover running vLLM servers
# ---------------------------------------------------------------------------
declare -a PIDS=()
declare -a MODELS=()
declare -a PORTS=()
declare -a JOBIDS=()
declare -a NODES=()

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    pid=$(echo "$line" | awk '{print $2}')
    # Extract --model value
    model=$(echo "$line" | grep -oP '(?<=--model )\S+')
    model_short=$(basename "$model")
    # Extract --port value
    port=$(echo "$line" | grep -oP '(?<=--port )\S+' || echo "8000")
    # Extract --jobid value
    jobid=$(echo "$line" | grep -oP '(?<=--jobid=)\S+' || echo "N/A")

    PIDS+=("$pid")
    MODELS+=("$model_short")
    PORTS+=("$port")
    JOBIDS+=("$jobid")
done < <(ps aux | grep '[v]llm.entrypoints.openai.api_server' | grep -v 'grep')

COUNT=${#PIDS[@]}

if [ "$COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No running vLLM servers found.${NC}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Display running servers
# ---------------------------------------------------------------------------
echo -e "${CYAN}==========================================${NC}"
echo -e "${BOLD}  Running vLLM Servers ($COUNT found)${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""
printf "  ${BOLD}%-4s %-8s %-30s %-8s %-12s${NC}\n" "#" "PID" "Model" "Port" "JobID"
printf "  %-4s %-8s %-30s %-8s %-12s\n" "----" "--------" "------------------------------" "--------" "------------"

for i in $(seq 0 $((COUNT - 1))); do
    printf "  %-4s %-8s %-30s %-8s %-12s\n" \
        "$((i + 1))" "${PIDS[$i]}" "${MODELS[$i]}" "${PORTS[$i]}" "${JOBIDS[$i]}"
done

echo ""
echo -e "  ${BOLD}0)${NC}  Stop ALL servers"
echo ""

# ---------------------------------------------------------------------------
# User selection
# ---------------------------------------------------------------------------
read -rp "Select server(s) to stop (e.g. 1, 1 3, or 0 for all): " selection

if [ -z "$selection" ]; then
    echo -e "${YELLOW}No selection made. Exiting.${NC}"
    exit 0
fi

# Build list of indices to kill
declare -a TO_KILL=()

if [ "$selection" = "0" ]; then
    for i in $(seq 0 $((COUNT - 1))); do
        TO_KILL+=("$i")
    done
else
    for s in $selection; do
        if [[ "$s" =~ ^[0-9]+$ ]] && [ "$s" -ge 1 ] && [ "$s" -le "$COUNT" ]; then
            TO_KILL+=("$((s - 1))")
        else
            echo -e "${RED}Invalid selection: $s (must be 1-$COUNT or 0)${NC}"
            exit 1
        fi
    done
fi

# ---------------------------------------------------------------------------
# Kill selected servers
# ---------------------------------------------------------------------------
for i in "${TO_KILL[@]}"; do
    pid="${PIDS[$i]}"
    model="${MODELS[$i]}"
    echo -ne "  Stopping ${BOLD}${model}${NC} (PID $pid) ... "

    # Kill the srun process and its children
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    echo -e "${GREEN}done${NC}"
done

echo ""
echo -e "${GREEN}Stopped ${#TO_KILL[@]} server(s).${NC}"
