#!/bin/bash
# =============================================================================
# Step 2: Start vLLM server with Qwen3.5-122B-A10B (MoE) on 4x A100 80GB
# Uses tensor parallelism (--tensor-parallel-size 4), single vLLM instance
#
# Usage: bash scripts/step-2-vLLM/step2_serve_122B.sh [--bf16]
#        Default: FP8 (recommended, ~122 GB weights, ~198 GB KV cache)
#        --bf16:  BF16 (~244 GB weights, ~76 GB KV cache)
#
# Prerequisites: run step1_alloc.sh first (need active salloc session)
#
# Memory budget (4x A100 80GB = 320 GB total):
#   FP8:  ~122 GB weights → ~198 GB for KV cache (recommended)
#   BF16: ~244 GB weights → ~76 GB for KV cache
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info_122B"
LOCAL_MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-122B-A10B"
PORT=8000
VLLM_IMAGE="vllm/vllm-openai:qwen3_5"
LOG="$WORKDIR/vllm_server_122B.log"
PID_FILE="$WORKDIR/.server_pid_122B"
TP_SIZE=4

# Parse args — default FP8
USE_BF16=false
if [[ "${1:-}" == "--bf16" ]]; then
    USE_BF16=true
fi

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Check prerequisites — read from .node_info if not in salloc shell
# ---------------------------------------------------------------------------
ALLOC_INFO="$WORKDIR/.node_info"
if [ -z "${SLURM_JOB_ID:-}" ] && [ -f "$ALLOC_INFO" ]; then
    source "$ALLOC_INFO"
    info "Read node info from $ALLOC_INFO"
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    error "No active SLURM allocation. Run step1_alloc.sh first."
fi

NODE="$SLURM_NODELIST"

echo -e "${CYAN}=========================================="
echo "  Starting vLLM: Qwen3.5-122B-A10B (MoE)"
echo "  Node: $NODE | Port: $PORT"
echo "  Tensor Parallel: $TP_SIZE GPUs"
if $USE_BF16; then
    echo "  Precision: BF16 (~244 GB weights)"
else
    echo "  Precision: FP8 (~122 GB weights)"
fi
echo "  Context: 128K tokens"
echo -e "==========================================${NC}"
echo ""

# ---------------------------------------------------------------------------
# Kill any existing server
# ---------------------------------------------------------------------------
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        warn "Stopping existing server (PID $OLD_PID) ..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

# Also kill anything on the port
pid=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$pid" ]; then
    warn "Killing existing process on port $PORT (PID $pid)"
    kill $pid 2>/dev/null || true
    sleep 2
fi

# ---------------------------------------------------------------------------
# Check model
# ---------------------------------------------------------------------------
SAFETENSOR_COUNT=$(find "$LOCAL_MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)

if [ "$SAFETENSOR_COUNT" -ge 1 ]; then
    MODEL="$LOCAL_MODEL_DIR"
    info "Using local model: $LOCAL_MODEL_DIR ($SAFETENSOR_COUNT shards)"
else
    error "Local model not found at $LOCAL_MODEL_DIR. Run download_Qwen3_5_122B.sh first."
fi

# ---------------------------------------------------------------------------
# Build vLLM launch command
# ---------------------------------------------------------------------------
VLLM_ARGS=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --language-model-only
    --tensor-parallel-size $TP_SIZE
    --max-model-len 131072
    --gpu-memory-utilization 0.90
    --port "$PORT"
    --host 0.0.0.0
    --dtype bfloat16
    --enforce-eager
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --enable-prefix-caching
    --max-num-seqs 256
)

if ! $USE_BF16; then
    VLLM_ARGS+=(--quantization fp8)
    info "Using FP8 quantization (~122 GB weights across $TP_SIZE GPUs)"
else
    info "Using BF16 full precision (~244 GB weights across $TP_SIZE GPUs)"
fi

# ---------------------------------------------------------------------------
# Launch vLLM server with all 4 GPUs (tensor parallelism)
# ---------------------------------------------------------------------------
info "Starting vLLM server on $NODE with TP=$TP_SIZE ..."
info "Log: $LOG"

srun --jobid="$SLURM_JOB_ID" -N 1 -n 1 --gpus $TP_SIZE \
    shifter --image="$VLLM_IMAGE" \
    "${VLLM_ARGS[@]}" \
    > "$LOG" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
info "Server process started (PID $SERVER_PID)"

# ---------------------------------------------------------------------------
# Wait for server to be ready (122B takes longer to load)
# ---------------------------------------------------------------------------
info "Waiting for model to load (~3-5 min for 122B) ..."
API_URL="http://${NODE}:${PORT}/v1"
MAX_WAIT=600
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if process died
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        error "Server process died. Last 50 lines of log:\n$(tail -50 "$LOG")"
    fi

    # Check if API is responding
    if curl -s --connect-timeout 3 "http://${NODE}:${PORT}/v1/models" &>/dev/null; then
        echo ""
        info "Server is ready!"
        echo ""

        # Show model info
        curl -s "${API_URL}/models" | python3 -m json.tool 2>/dev/null || true
        echo ""

        # Show GPU usage
        srun --jobid="$SLURM_JOB_ID" -N 1 -n 1 --overlap --gpus $TP_SIZE bash -c "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null || true
        echo ""

        # Save node info
        cat > "$NODE_INFO" <<EOF
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODELIST=$SLURM_NODELIST
ALLOCATED_AT="$(date "+%Y-%m-%d %H:%M:%S")"
API_URL=$API_URL
SERVER_PID=$SERVER_PID
MODEL=$MODEL
TP_SIZE=$TP_SIZE
EOF

        echo -e "${GREEN}=========================================="
        echo "  vLLM Server Ready! (122B-A10B)"
        echo "  API:  $API_URL"
        echo "  Node: $NODE"
        echo "  TP:   $TP_SIZE GPUs"
        echo "  PID:  $SERVER_PID"
        echo "  Log:  $LOG"
        echo ""
        echo "  Test: bash scripts/step-3-test-agentic/test_tool_demo_122B.sh"
        echo "  Stop: kill $SERVER_PID"
        echo -e "==========================================${NC}"
        exit 0
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    printf "\r  Waiting ... ${ELAPSED}s "
done

echo ""
warn "Server did not respond within ${MAX_WAIT}s."
warn "Check log: tail -f $LOG"
warn "Server PID: $SERVER_PID"
exit 1
