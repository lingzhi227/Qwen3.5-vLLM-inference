#!/bin/bash
# =============================================================================
# Step 1: Allocate 2x A100 80GB GPU nodes on Perlmutter (single salloc)
# Usage: bash scripts/step-1-salloc/start_alloc_2.sh
#
# Allocates 2 nodes in one job. Saves .node_info and .node_info_2
# for compatibility with all start_serve scripts.
#
# Then run: bash scripts/step-2-vLLM/start_serve_multi_2.sh
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO_1="$WORKDIR/.node_info"
NODE_INFO_2="$WORKDIR/.node_info_2"

# Colors
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}=========================================="
echo "  Requesting 2x A100 80GB GPU Nodes"
echo "  Queue: interactive | Time: 4h"
echo "  Account: m5242_g"
echo -e "==========================================${NC}"
echo ""

# Clean up old node info
rm -f "$NODE_INFO_1" "$NODE_INFO_2"

salloc -N 2 -q interactive -t 04:00:00 -C "gpu&hbm80g" -A m5242_g \
  bash -c '
    WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
    NODE_INFO_1="$WORKDIR/.node_info"
    NODE_INFO_2="$WORKDIR/.node_info_2"

    # Parse node list (e.g. "nid[200332,200272]" or "nid200332,nid200272")
    NODELIST="$SLURM_NODELIST"
    # Expand compressed nodelist using scontrol
    EXPANDED=$(scontrol show hostnames "$NODELIST" 2>/dev/null)
    NODE_1=$(echo "$EXPANDED" | head -1)
    NODE_2=$(echo "$EXPANDED" | tail -1)

    if [ -z "$NODE_1" ] || [ -z "$NODE_2" ] || [ "$NODE_1" = "$NODE_2" ]; then
        echo -e "\033[0;31m[ERROR] Failed to get 2 distinct nodes from: $NODELIST\033[0m"
        exit 1
    fi

    # Save node info (both files point to same job, different nodes)
    cat > "$NODE_INFO_1" <<EOF
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODELIST=$NODE_1
ALLOCATED_AT="$(date "+%Y-%m-%d %H:%M:%S")"
EOF

    cat > "$NODE_INFO_2" <<EOF
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODELIST=$NODE_2
ALLOCATED_AT="$(date "+%Y-%m-%d %H:%M:%S")"
EOF

    echo ""
    echo -e "\033[0;32m============================================"
    echo "  2x GPU Nodes Allocated!"
    echo "  Job ID:  $SLURM_JOB_ID"
    echo "  Node 1:  $NODE_1"
    echo "  Node 2:  $NODE_2"
    echo "  Saved:   $NODE_INFO_1"
    echo "           $NODE_INFO_2"
    echo -e "============================================\033[0m"
    echo ""

    # Verify GPUs on both nodes
    echo "Verifying GPUs on $NODE_1 ..."
    srun -N 1 -n 1 --gpus 4 --nodelist="$NODE_1" \
        bash -c "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" 2>/dev/null || echo "(GPU check skipped)"
    echo ""
    echo "Verifying GPUs on $NODE_2 ..."
    srun -N 1 -n 1 --gpus 4 --nodelist="$NODE_2" \
        bash -c "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" 2>/dev/null || echo "(GPU check skipped)"
    echo ""

    echo "Next step:"
    echo "  cd $WORKDIR && bash scripts/step-2-vLLM/start_serve_multi_2.sh"
    echo ""

    # Drop into interactive shell
    exec bash
  '
