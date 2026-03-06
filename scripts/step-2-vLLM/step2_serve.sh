#!/bin/bash
# =============================================================================
# Step 2: Start vLLM server with Qwen3.5-27B on the allocated GPU node
# Usage: bash scripts/step2_serve.sh [--fp8]
#
# Prerequisites: run step1_alloc.sh first (need active salloc session)
#
# Memory budget (A100 80GB, single GPU, 10-15 concurrent agents):
#   BF16 weights:  ~55.6 GB
#   KV cache:      ~16 GB available (only 16/64 layers are full attention)
#                  = ~256K tokens total across all agents
#                  = ~16K tokens/agent with 15 concurrent agents
#   For tool-calling agents (~4-8K tokens each), BF16 fits comfortably.
#
# If BF16 OOMs, re-run with: bash scripts/step2_serve.sh --fp8
#   FP8 weights:   ~28 GB → ~44 GB for KV → ~688K tokens total
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info"
MODEL_HF="Qwen/Qwen3.5-27B"
LOCAL_MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-27B"
PORT=8000
VLLM_IMAGE="vllm/vllm-openai:qwen3_5"
LOG="$WORKDIR/vllm_server.log"
PID_FILE="$WORKDIR/.server_pid"

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
# Check prerequisites
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    error "No active SLURM allocation. Run step1_alloc.sh first."
fi

if [ ! -f "$NODE_INFO" ]; then
    # Recreate from env
    cat > "$NODE_INFO" <<EOF
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODELIST=$SLURM_NODELIST
ALLOCATED_AT="$(date "+%Y-%m-%d %H:%M:%S")"
EOF
fi

source "$NODE_INFO"
NODE="$SLURM_NODELIST"

echo -e "${CYAN}=========================================="
echo "  Starting vLLM: Qwen3.5-27B"
echo "  Node: $NODE | Port: $PORT"
if $USE_FP8; then
    echo "  Precision: FP8 (quantized)"
else
    echo "  Precision: BF16 (full)"
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

# ---------------------------------------------------------------------------
# Resolve model path: prefer local pscratch, fallback to HF hub name
# ---------------------------------------------------------------------------
SAFETENSOR_COUNT=$(find "$LOCAL_MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)

if [ "$SAFETENSOR_COUNT" -ge 11 ]; then
    MODEL="$LOCAL_MODEL_DIR"
    info "Using local model: $LOCAL_MODEL_DIR ($SAFETENSOR_COUNT shards)"
else
    MODEL="$MODEL_HF"
    MODEL="$MODEL_HF"
    warn "Local model not found at $LOCAL_MODEL_DIR ($SAFETENSOR_COUNT/11 shards)"
    warn "Will download from HuggingFace on first run. To pre-download:"
    warn "  bash scripts/download_model.sh"
    echo ""
fi

# ---------------------------------------------------------------------------
# Build vLLM launch command
# ---------------------------------------------------------------------------
VLLM_ARGS=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --language-model-only
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

if $USE_FP8; then
    VLLM_ARGS+=(--quantization fp8)
    info "Using FP8 quantization (~28 GB weights)"
else
    info "Using BF16 full precision (~55.6 GB weights)"
fi

# ---------------------------------------------------------------------------
# Launch vLLM server
# ---------------------------------------------------------------------------
info "Starting vLLM server on $NODE ..."
info "Log: $LOG"

srun -N 1 -n 1 --gpus 1 \
    shifter --image="$VLLM_IMAGE" \
    "${VLLM_ARGS[@]}" \
    > "$LOG" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
info "Server process started (PID $SERVER_PID)"

# ---------------------------------------------------------------------------
# Wait for server to be ready
# ---------------------------------------------------------------------------
info "Waiting for model to load (may take 3-10 minutes for first download) ..."
API_URL="http://${NODE}:${PORT}/v1"
MAX_WAIT=600  # 10 minutes
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if process died
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        error "Server process died. Last 30 lines of log:\n$(tail -30 "$LOG")"
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
        srun -N 1 -n 1 --overlap bash -c "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null || true
        echo ""

        # Save API URL
        echo "API_URL=$API_URL" >> "$NODE_INFO"
        echo "SERVER_PID=$SERVER_PID" >> "$NODE_INFO"

        echo -e "${GREEN}=========================================="
        echo "  vLLM Server Ready!"
        echo "  API:  $API_URL"
        echo "  Node: $NODE"
        echo "  PID:  $SERVER_PID"
        echo "  Log:  $LOG"
        echo ""
        echo "  Test: bash scripts/test_tool_demo.sh"
        echo "  Stop: kill $SERVER_PID"
        echo -e "==========================================${NC}"
        exit 0
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    printf "."
done

echo ""
warn "Server did not respond within ${MAX_WAIT}s."
warn "Check log: tail -f $LOG"
warn "Server PID: $SERVER_PID"
exit 1
