#!/bin/bash
# =============================================================================
# Step 2 (Multi-GPU): Start 4 vLLM servers on all 4 A100 GPUs
# Each server runs on a separate GPU and port (8000-8003)
#
# Usage: bash scripts/step-2-vLLM/step2_serve_multi.sh [--fp8]
# Prerequisites: run step-1-salloc/start_alloc.sh first (need active salloc session)
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info_multi"
LOCAL_MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-27B"
VLLM_IMAGE="vllm/vllm-openai:qwen3_5"
NUM_GPUS=4
BASE_PORT=8000

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

# Check prerequisites — read from .node_info if not in salloc shell
ALLOC_INFO="$WORKDIR/.node_info"
if [ -z "${SLURM_JOB_ID:-}" ] && [ -f "$ALLOC_INFO" ]; then
    source "$ALLOC_INFO"
    info "Read node info from $ALLOC_INFO"
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    error "No active SLURM allocation. Run step-1-salloc/start_alloc.sh first."
fi

NODE="$SLURM_NODELIST"

echo -e "${CYAN}=========================================="
echo "  Starting 4x vLLM: Qwen3.5-27B"
echo "  Node: $NODE | Ports: ${BASE_PORT}-$((BASE_PORT+NUM_GPUS-1))"
if $USE_FP8; then
    echo "  Precision: FP8 (quantized)"
else
    echo "  Precision: BF16 (full)"
fi
echo "  Context: 256K tokens"
echo -e "==========================================${NC}"
echo ""

# Kill any existing servers on these ports
for port in $(seq $BASE_PORT $((BASE_PORT+NUM_GPUS-1))); do
    pid=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        warn "Killing existing process on port $port (PID $pid)"
        kill $pid 2>/dev/null || true
    fi
done
sleep 2

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
# Launch 4 vLLM servers via a single srun with all 4 GPUs
# Each server gets CUDA_VISIBLE_DEVICES=N and a unique port
# ---------------------------------------------------------------------------
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((BASE_PORT + gpu_id))
    log="$WORKDIR/vllm_server_gpu${gpu_id}.log"

    info "Starting GPU $gpu_id on port $port (log: $log)"

    srun --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 --gpus=4 \
        shifter --image="$VLLM_IMAGE" \
        bash -c "CUDA_VISIBLE_DEVICES=$gpu_id python3 -m vllm.entrypoints.openai.api_server ${COMMON_ARGS[*]} --port $port" \
        > "$log" 2>&1 &

    PIDS+=($!)
done

info "All 4 servers starting (PIDs: ${PIDS[*]})"

# ---------------------------------------------------------------------------
# Wait for all servers to be ready
# ---------------------------------------------------------------------------
info "Waiting for servers to load model (~80s) ..."
MAX_WAIT=600
ELAPSED=0
INTERVAL=5
READY_COUNT=0

while [ $ELAPSED -lt $MAX_WAIT ] && [ $READY_COUNT -lt $NUM_GPUS ]; do
    READY_COUNT=0
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        port=$((BASE_PORT + gpu_id))
        if curl -s --connect-timeout 3 "http://${NODE}:${port}/v1/models" &>/dev/null; then
            READY_COUNT=$((READY_COUNT+1))
        fi
    done

    if [ $READY_COUNT -lt $NUM_GPUS ]; then
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        printf "\r  Ready: ${READY_COUNT}/${NUM_GPUS} (${ELAPSED}s) "
    fi
done
echo ""

if [ $READY_COUNT -lt $NUM_GPUS ]; then
    warn "Only ${READY_COUNT}/${NUM_GPUS} servers ready after ${MAX_WAIT}s"
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        port=$((BASE_PORT + gpu_id))
        if ! curl -s --connect-timeout 3 "http://${NODE}:${port}/v1/models" &>/dev/null; then
            warn "  GPU $gpu_id (port $port) NOT ready. Check: tail $WORKDIR/vllm_server_gpu${gpu_id}.log"
        fi
    done
    if [ $READY_COUNT -eq 0 ]; then
        error "No servers started. Check logs."
    fi
fi

# Save node info
API_URLS=""
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((BASE_PORT + gpu_id))
    if [ -n "$API_URLS" ]; then API_URLS="$API_URLS,"; fi
    API_URLS="${API_URLS}http://${NODE}:${port}/v1"
done

cat > "$NODE_INFO" <<EOF
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODELIST=$SLURM_NODELIST
ALLOCATED_AT="$(date "+%Y-%m-%d %H:%M:%S")"
NUM_GPUS=$NUM_GPUS
BASE_PORT=$BASE_PORT
API_URLS="$API_URLS"
SERVER_PIDS="${PIDS[*]}"
EOF

echo -e "${GREEN}=========================================="
echo "  ${READY_COUNT}x vLLM Servers Ready!"
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((BASE_PORT + gpu_id))
    echo "  GPU $gpu_id: http://${NODE}:${port}/v1"
done
echo ""
echo "  Test: bash scripts/step-4-test-stress/test_stress_multi.sh"
echo "  Stop: kill ${PIDS[*]}"
echo -e "==========================================${NC}"
