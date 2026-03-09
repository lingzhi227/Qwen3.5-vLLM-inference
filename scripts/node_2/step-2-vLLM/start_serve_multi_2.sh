#!/bin/bash
# =============================================================================
# Step 2 (2-Node, 8 GPUs): Start 8 vLLM servers across 2 A100 GPU nodes
# Node 1 (from .node_info):   GPUs 0-3, ports 8000-8003
# Node 2 (from .node_info_2): GPUs 0-3, ports 8004-8007
#
# Usage: bash scripts/step-2-vLLM/start_serve_multi_2.sh [--fp8]
# Prerequisites: run start_alloc.sh AND start_alloc_2.sh first
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO_OUT="$WORKDIR/.node_info_multi_2"
LOCAL_MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-27B"
VLLM_IMAGE="vllm/vllm-openai:qwen3_5"
NUM_GPUS_PER_NODE=4
BASE_PORT_NODE1=8000
BASE_PORT_NODE2=8004

# Parse args
USE_FP8=false
if [[ "${1:-}" == "--fp8" ]]; then
    USE_FP8=true
fi

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Read node info from both allocations
# ---------------------------------------------------------------------------
ALLOC_INFO_1="$WORKDIR/.node_info"
ALLOC_INFO_2="$WORKDIR/.node_info_2"

[ -f "$ALLOC_INFO_1" ] || error "Node 1 info not found ($ALLOC_INFO_1). Run start_alloc.sh first."
[ -f "$ALLOC_INFO_2" ] || error "Node 2 info not found ($ALLOC_INFO_2). Run start_alloc_2.sh first."

# Source node 1
JOB_ID_1=$(grep SLURM_JOB_ID "$ALLOC_INFO_1" | cut -d= -f2)
NODE_1=$(grep SLURM_NODELIST "$ALLOC_INFO_1" | cut -d= -f2)

# Source node 2
JOB_ID_2=$(grep SLURM_JOB_ID "$ALLOC_INFO_2" | cut -d= -f2)
NODE_2=$(grep SLURM_NODELIST "$ALLOC_INFO_2" | cut -d= -f2)

[ -n "$JOB_ID_1" ] || error "Could not read JOB_ID from $ALLOC_INFO_1"
[ -n "$JOB_ID_2" ] || error "Could not read JOB_ID from $ALLOC_INFO_2"

echo -e "${CYAN}=========================================="
echo "  Starting 8x vLLM: Qwen3.5-27B (2 nodes)"
echo "  Node 1: $NODE_1 (Job $JOB_ID_1) | Ports: ${BASE_PORT_NODE1}-$((BASE_PORT_NODE1+NUM_GPUS_PER_NODE-1))"
echo "  Node 2: $NODE_2 (Job $JOB_ID_2) | Ports: ${BASE_PORT_NODE2}-$((BASE_PORT_NODE2+NUM_GPUS_PER_NODE-1))"
if $USE_FP8; then
    echo "  Precision: FP8 (quantized)"
else
    echo "  Precision: BF16 (full)"
fi
echo "  Context: 128K tokens"
echo -e "==========================================${NC}"
echo ""

# Check model
SAFETENSOR_COUNT=$(find "$LOCAL_MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
if [ "$SAFETENSOR_COUNT" -ge 11 ]; then
    MODEL="$LOCAL_MODEL_DIR"
    info "Using local model: $LOCAL_MODEL_DIR ($SAFETENSOR_COUNT shards)"
else
    error "Local model not found at $LOCAL_MODEL_DIR ($SAFETENSOR_COUNT/11 shards). Run download_model.sh first."
fi

# Build common vLLM args
COMMON_ARGS=(
    --model "$MODEL"
    --language-model-only
    --max-model-len 262144
    --gpu-memory-utilization 0.90
    --host 0.0.0.0
    --dtype bfloat16
    --enforce-eager
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --enable-prefix-caching
    --max-num-seqs 256
)

if $USE_FP8; then
    COMMON_ARGS+=(--quantization fp8)
fi

# ---------------------------------------------------------------------------
# Launch servers on both nodes
# ---------------------------------------------------------------------------
PIDS=()

# --- Node 1: GPUs 0-3 on ports 8000-8003 ---
for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
    port=$((BASE_PORT_NODE1 + gpu_id))
    log="$WORKDIR/vllm_server_n1_gpu${gpu_id}.log"

    info "Node 1 ($NODE_1): Starting GPU $gpu_id on port $port (log: $log)"

    srun --jobid="$JOB_ID_1" --nodelist="$NODE_1" --overlap -N 1 -n 1 --gpus=4 \
        shifter --image="$VLLM_IMAGE" \
        bash -c "CUDA_VISIBLE_DEVICES=$gpu_id python3 -m vllm.entrypoints.openai.api_server ${COMMON_ARGS[*]} --port $port" \
        > "$log" 2>&1 &

    PIDS+=($!)
done

# --- Node 2: GPUs 0-3 on ports 8004-8007 ---
for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
    port=$((BASE_PORT_NODE2 + gpu_id))
    log="$WORKDIR/vllm_server_n2_gpu${gpu_id}.log"

    info "Node 2 ($NODE_2): Starting GPU $gpu_id on port $port (log: $log)"

    srun --jobid="$JOB_ID_2" --nodelist="$NODE_2" --overlap -N 1 -n 1 --gpus=4 \
        shifter --image="$VLLM_IMAGE" \
        bash -c "CUDA_VISIBLE_DEVICES=$gpu_id python3 -m vllm.entrypoints.openai.api_server ${COMMON_ARGS[*]} --port $port" \
        > "$log" 2>&1 &

    PIDS+=($!)
done

TOTAL_GPUS=$((NUM_GPUS_PER_NODE * 2))
info "All $TOTAL_GPUS servers starting (PIDs: ${PIDS[*]})"

# ---------------------------------------------------------------------------
# Wait for all servers to be ready
# ---------------------------------------------------------------------------
info "Waiting for servers to load model (~80s) ..."
MAX_WAIT=600
ELAPSED=0
INTERVAL=5
READY_COUNT=0

while [ $ELAPSED -lt $MAX_WAIT ] && [ $READY_COUNT -lt $TOTAL_GPUS ]; do
    READY_COUNT=0
    # Check node 1
    for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
        port=$((BASE_PORT_NODE1 + gpu_id))
        if curl -s --connect-timeout 3 "http://${NODE_1}:${port}/v1/models" &>/dev/null; then
            READY_COUNT=$((READY_COUNT+1))
        fi
    done
    # Check node 2
    for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
        port=$((BASE_PORT_NODE2 + gpu_id))
        if curl -s --connect-timeout 3 "http://${NODE_2}:${port}/v1/models" &>/dev/null; then
            READY_COUNT=$((READY_COUNT+1))
        fi
    done

    if [ $READY_COUNT -lt $TOTAL_GPUS ]; then
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        printf "\r  Ready: ${READY_COUNT}/${TOTAL_GPUS} (${ELAPSED}s) "
    fi
done
echo ""

if [ $READY_COUNT -lt $TOTAL_GPUS ]; then
    warn "Only ${READY_COUNT}/${TOTAL_GPUS} servers ready after ${MAX_WAIT}s"
    for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
        port=$((BASE_PORT_NODE1 + gpu_id))
        if ! curl -s --connect-timeout 3 "http://${NODE_1}:${port}/v1/models" &>/dev/null; then
            warn "  Node 1 GPU $gpu_id (port $port) NOT ready. Check: tail $WORKDIR/vllm_server_n1_gpu${gpu_id}.log"
        fi
    done
    for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
        port=$((BASE_PORT_NODE2 + gpu_id))
        if ! curl -s --connect-timeout 3 "http://${NODE_2}:${port}/v1/models" &>/dev/null; then
            warn "  Node 2 GPU $gpu_id (port $port) NOT ready. Check: tail $WORKDIR/vllm_server_n2_gpu${gpu_id}.log"
        fi
    done
    if [ $READY_COUNT -eq 0 ]; then
        error "No servers started. Check logs."
    fi
fi

# Save node info
API_URLS=""
for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
    port=$((BASE_PORT_NODE1 + gpu_id))
    if [ -n "$API_URLS" ]; then API_URLS="$API_URLS,"; fi
    API_URLS="${API_URLS}http://${NODE_1}:${port}/v1"
done
for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
    port=$((BASE_PORT_NODE2 + gpu_id))
    API_URLS="${API_URLS},http://${NODE_2}:${port}/v1"
done

cat > "$NODE_INFO_OUT" <<EOF
SLURM_JOB_ID_1=$JOB_ID_1
SLURM_JOB_ID_2=$JOB_ID_2
SLURM_NODE_1=$NODE_1
SLURM_NODE_2=$NODE_2
ALLOCATED_AT="$(date "+%Y-%m-%d %H:%M:%S")"
NUM_GPUS=$TOTAL_GPUS
BASE_PORT_NODE1=$BASE_PORT_NODE1
BASE_PORT_NODE2=$BASE_PORT_NODE2
API_URLS="$API_URLS"
SERVER_PIDS="${PIDS[*]}"
EOF

echo -e "${GREEN}=========================================="
echo "  ${READY_COUNT}x vLLM Servers Ready!"
echo ""
echo "  Node 1 ($NODE_1):"
for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
    port=$((BASE_PORT_NODE1 + gpu_id))
    echo "    GPU $gpu_id: http://${NODE_1}:${port}/v1"
done
echo ""
echo "  Node 2 ($NODE_2):"
for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE-1))); do
    port=$((BASE_PORT_NODE2 + gpu_id))
    echo "    GPU $gpu_id: http://${NODE_2}:${port}/v1"
done
echo ""
echo "  Info saved: $NODE_INFO_OUT"
echo "  Stop: kill ${PIDS[*]}"
echo -e "==========================================${NC}"
