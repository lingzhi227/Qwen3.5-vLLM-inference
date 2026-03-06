#!/bin/bash
# =============================================================================
# Step 1: Allocate an A100 80GB GPU node on Perlmutter
# Usage: bash step1_alloc.sh
#
# After allocation, node info is saved and you get an interactive shell.
# Then run: bash scripts/step2_serve.sh
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info"

# Colors
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}=========================================="
echo "  Requesting A100 80GB GPU Node"
echo "  Queue: interactive | Time: 4h"
echo "  Account: m5242_g"
echo -e "==========================================${NC}"
echo ""

# Clean up old node info
rm -f "$NODE_INFO"

salloc -N 1 -q interactive -t 04:00:00 -C "gpu&hbm80g" -A m5242_g \
  bash -c '
    WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
    NODE_INFO="$WORKDIR/.node_info"

    # Save node info
    cat > "$NODE_INFO" <<EOF
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODELIST=$SLURM_NODELIST
ALLOCATED_AT="$(date "+%Y-%m-%d %H:%M:%S")"
EOF

    echo ""
    echo -e "\033[0;32m============================================"
    echo "  GPU Node Allocated!"
    echo "  Job ID:  $SLURM_JOB_ID"
    echo "  Node:    $SLURM_NODELIST"
    echo "  Saved:   $NODE_INFO"
    echo -e "============================================\033[0m"
    echo ""

    # Verify GPU availability
    echo "Verifying GPUs on $SLURM_NODELIST ..."
    srun -N 1 -n 1 --gpus 4 bash -c "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" 2>/dev/null || echo "(GPU check skipped)"
    echo ""
    echo "Next step:"
    echo "  cd $WORKDIR && bash scripts/step2_serve.sh"
    echo ""

    # Drop into interactive shell (inherits SLURM env vars)
    exec bash
  '
