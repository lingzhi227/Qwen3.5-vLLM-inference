#!/bin/bash
# =============================================================================
# Start vLLM server with Qwen3.5-27B-AWQ-4bit on A100-PCIE-40GB
# AWQ 4-bit (~20GB) + BF16 KV cache (~16GB @ 256K) = ~38GB — tight on 40GB
#
# Memory budget (40GB × 0.95 = 38GB usable):
#   Model weights (AWQ 4-bit): ~20 GB
#   KV cache (BF16, 16 attn layers, 256K): ~16 GB
#   Overhead (CUDA ctx + activations): ~2 GB
#   Total: ~38 GB — fits but barely, batch_size=1 only
#
# If OOM at 256K, try: bash serve.sh --max-len 131072
#
# Usage: bash scripts/PCIE_27B_INT4/serve.sh [--max-len N] [--port N]
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-27B-AWQ-4bit"
VLLM_IMAGE="vllm/vllm-openai:qwen3_5"
PORT=8000
LOG="$WORKDIR/vllm_pcie_27b_int4.log"
NODE_INFO="$WORKDIR/.node_info_pcie_27b_int4"

# Defaults
DTYPE="bfloat16"
MAX_MODEL_LEN=262144

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-len)
            MAX_MODEL_LEN="$2"
            shift 2 ;;
        --port)
            PORT="$2"
            shift 2 ;;
        *)
            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo -e "${CYAN}=========================================="
echo "  vLLM: Qwen3.5-27B-AWQ-4bit (cyankiwi)"
echo "  GPU:  A100-PCIE-40GB"
echo "  Port: $PORT | Context: ${MAX_MODEL_LEN} tokens"
echo "  Memory: ~20GB model + ~16GB KV = ~38GB"
echo -e "==========================================${NC}"
echo ""

# Check GPU
if ! nvidia-smi &>/dev/null; then
    error "No GPU found on this node"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
info "GPU: $GPU_NAME (${GPU_MEM}MiB)"

if [ "${GPU_MEM:-0}" -lt 38000 ]; then
    warn "GPU memory ${GPU_MEM}MiB may be insufficient. Need ~38GB for 27B-AWQ-4bit + 256K context."
    warn "Try: bash serve.sh --max-len 131072  (reduces KV to ~8GB)"
fi

# Check model
if [ ! -d "$MODEL_DIR" ] || [ "$(find "$MODEL_DIR" -name '*.safetensors' 2>/dev/null | wc -l)" -eq 0 ]; then
    error "Model not found at $MODEL_DIR. Run download.sh first."
fi
info "Model: $MODEL_DIR"

# Kill existing server on this port
pid=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$pid" ]; then
    warn "Killing existing process on port $PORT (PID $pid)"
    kill $pid 2>/dev/null || true
    sleep 2
fi

# Build vLLM args
# - No --quantization flag: auto-detected as "compressed-tensors" from config.json
#   (DeltaNet in_proj_a/b kept at full precision, rest INT4 group_size=32)
# - gpu-memory-utilization 0.95: maximize KV cache space (need every GB)
# - enable-chunked-prefill: process long prompts in chunks (avoids OOM during prefill)
# - max-num-batched-tokens 4096: limit prefill batch to avoid memory spikes
VLLM_ARGS=(
    --model "$MODEL_DIR"
    --language-model-only
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization 0.95
    --host 0.0.0.0
    --port "$PORT"
    --dtype "$DTYPE"
    --enforce-eager
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --enable-prefix-caching
    --max-num-seqs 256
)

info "Starting vLLM server (log: $LOG)..."
info "Args: ${VLLM_ARGS[*]}"

# Launch with shifter
CUDA_VISIBLE_DEVICES=0 shifter --image="$VLLM_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
    ${VLLM_ARGS[@]} \
    > "$LOG" 2>&1 &

SERVER_PID=$!
info "Server PID: $SERVER_PID"

# Wait for server to be ready
info "Waiting for server to load model (~60s for AWQ INT4)..."
MAX_WAIT=300
ELAPSED=0
INTERVAL=5
HOSTNAME=$(hostname)

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        error "Server process died. Check log: tail -50 $LOG"
    fi

    if curl -s --connect-timeout 3 "http://localhost:${PORT}/v1/models" &>/dev/null; then
        echo ""
        info "Server ready!"
        break
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    printf "\r  Loading... (${ELAPSED}s) "
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    error "Server failed to start in ${MAX_WAIT}s. Check: tail -50 $LOG"
fi

# Save node info
API_URL="http://${HOSTNAME}:${PORT}/v1"
cat > "$NODE_INFO" <<EOF
API_URL="$API_URL"
SERVER_PID=$SERVER_PID
MODEL_DIR=$MODEL_DIR
HOSTNAME=$HOSTNAME
PORT=$PORT
MAX_MODEL_LEN=$MAX_MODEL_LEN
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"
EOF

# Verify with a quick query
MODEL_NAME=$(curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

echo -e "${GREEN}=========================================="
echo "  vLLM Server Ready!"
echo ""
echo "  Model:   $MODEL_NAME (AWQ 4-bit)"
echo "  URL:     $API_URL"
echo "  PID:     $SERVER_PID"
echo "  Context: $MAX_MODEL_LEN tokens"
echo ""
echo "  Test:    bash scripts/PCIE_27B_INT4/test_cwl_agent.sh"
echo "  BixBench: bash scripts/PCIE_27B_INT4/test_bixbench.sh"
echo "  Log:     tail -f $LOG"
echo "  Stop:    kill $SERVER_PID"
echo -e "==========================================${NC}"
