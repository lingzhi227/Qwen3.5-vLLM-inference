#!/bin/bash
# =============================================================================
# Stop GPU allocations interactively
# Lists all active SLURM jobs and lets you choose which to cancel.
# Usage: bash scripts/step-1-salloc/stop_alloc.sh
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
BOLD='\033[1m'

# ---------------------------------------------------------------------------
# Discover active jobs (RUNNING + PENDING)
# ---------------------------------------------------------------------------
declare -a JOBIDS=()
declare -a NAMES=()
declare -a PARTITIONS=()
declare -a STATES=()
declare -a NODES=()
declare -a TIMES=()
declare -a GPUS=()

while IFS='|' read -r jobid name partition state nodelist time tres; do
    [[ -z "$jobid" || "$jobid" == "JOBID" ]] && continue
    # Trim whitespace
    jobid=$(echo "$jobid" | xargs)
    name=$(echo "$name" | xargs)
    partition=$(echo "$partition" | xargs)
    state=$(echo "$state" | xargs)
    nodelist=$(echo "$nodelist" | xargs)
    time=$(echo "$time" | xargs)
    tres=$(echo "$tres" | xargs)

    JOBIDS+=("$jobid")
    NAMES+=("$name")
    PARTITIONS+=("$partition")
    STATES+=("$state")
    NODES+=("$nodelist")
    TIMES+=("$time")
    GPUS+=("$tres")
done < <(squeue -u "$USER" --states=RUNNING,PENDING --format="%i|%j|%P|%T|%N|%M|%b" 2>/dev/null)

COUNT=${#JOBIDS[@]}

if [ "$COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No active SLURM jobs found.${NC}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Display active jobs
# ---------------------------------------------------------------------------
echo -e "${CYAN}==========================================${NC}"
echo -e "${BOLD}  Active SLURM Jobs ($COUNT found)${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""
printf "  ${BOLD}%-4s %-12s %-20s %-10s %-14s %-12s %-10s${NC}\n" \
    "#" "JobID" "Name" "State" "Node" "Time" "GPU"
printf "  %-4s %-12s %-20s %-10s %-14s %-12s %-10s\n" \
    "----" "------------" "--------------------" "----------" "--------------" "------------" "----------"

for i in $(seq 0 $((COUNT - 1))); do
    printf "  %-4s %-12s %-20s %-10s %-14s %-12s %-10s\n" \
        "$((i + 1))" "${JOBIDS[$i]}" "${NAMES[$i]}" "${STATES[$i]}" "${NODES[$i]}" "${TIMES[$i]}" "${GPUS[$i]}"
done

echo ""
echo -e "  ${BOLD}0)${NC}  Cancel ALL jobs"
echo ""

# ---------------------------------------------------------------------------
# User selection
# ---------------------------------------------------------------------------
read -rp "Select job(s) to cancel (e.g. 1, 1 3, or 0 for all): " selection

if [ -z "$selection" ]; then
    echo -e "${YELLOW}No selection made. Exiting.${NC}"
    exit 0
fi

# Build list of indices to cancel
declare -a TO_CANCEL=()

if [ "$selection" = "0" ]; then
    for i in $(seq 0 $((COUNT - 1))); do
        TO_CANCEL+=("$i")
    done
else
    for s in $selection; do
        if [[ "$s" =~ ^[0-9]+$ ]] && [ "$s" -ge 1 ] && [ "$s" -le "$COUNT" ]; then
            TO_CANCEL+=("$((s - 1))")
        else
            echo -e "${RED}Invalid selection: $s (must be 1-$COUNT or 0)${NC}"
            exit 1
        fi
    done
fi

# ---------------------------------------------------------------------------
# Cancel selected jobs
# ---------------------------------------------------------------------------
for i in "${TO_CANCEL[@]}"; do
    jobid="${JOBIDS[$i]}"
    name="${NAMES[$i]}"
    echo -ne "  Cancelling ${BOLD}${name}${NC} (JobID $jobid) ... "
    scancel "$jobid" 2>/dev/null
    echo -e "${GREEN}done${NC}"
done

echo ""
echo -e "${GREEN}Cancelled ${#TO_CANCEL[@]} job(s).${NC}"
