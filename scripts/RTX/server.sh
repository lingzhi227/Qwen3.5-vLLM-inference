#!/usr/bin/env bash
# =============================================================================
# vLLM Server Management Script (runs on the remote GPU machine)
# Usage: bash server.sh [setup|start|stop|status|logs]
# =============================================================================
set -euo pipefail

WORKDIR="$HOME/qwen-server"
MODEL="Qwen/Qwen3.5-9B"
PORT=8000
LOG="$WORKDIR/server.log"
PID_FILE="$WORKDIR/server.pid"

# --- Colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --- Find CUDA 12 libs (system may have CUDA 13, vLLM needs 12) ---
setup_cuda_env() {
    local venv_lib="$WORKDIR/.venv/lib/python3*/site-packages/nvidia"
    local cuda_dirs=()
    for d in $venv_lib/*/lib; do
        [ -d "$d" ] && cuda_dirs+=("$d")
    done
    if [ ${#cuda_dirs[@]} -gt 0 ]; then
        export LD_LIBRARY_PATH=$(IFS=:; echo "${cuda_dirs[*]}"):${LD_LIBRARY_PATH:-}
        info "LD_LIBRARY_PATH set with ${#cuda_dirs[@]} NVIDIA lib dirs"
    fi
}

# --- One-time setup ---
do_setup() {
    info "Setting up vLLM environment at $WORKDIR ..."

    # Install uv if missing
    if ! command -v uv &>/dev/null; then
        info "Installing uv ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Create venv
    mkdir -p "$WORKDIR"
    cd "$WORKDIR"
    if [ ! -d ".venv" ]; then
        info "Creating Python venv ..."
        uv venv .venv
    fi
    source .venv/bin/activate

    # Install vLLM + deps
    info "Installing vLLM and dependencies ..."
    uv pip install \
        vllm \
        bitsandbytes \
        nvidia-cuda-runtime-cu12 \
        nvidia-cublas-cu12 \
        nvidia-cudnn-cu12

    info "Setup complete!"
    echo ""
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
}

# --- Start server ---
do_start() {
    cd "$WORKDIR"

    # Check if already running
    if [ -f "$PID_FILE" ]; then
        local old_pid
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            warn "Server already running (PID $old_pid). Use 'stop' first."
            return 0
        fi
        rm -f "$PID_FILE"
    fi

    source .venv/bin/activate
    setup_cuda_env

    info "Starting vLLM server ..."
    info "Model: $MODEL"
    info "Port:  $PORT"
    info "Log:   $LOG"

    nohup vllm serve "$MODEL" \
        --language-model-only \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --max-model-len 15000 \
        --gpu-memory-utilization 0.9 \
        --port "$PORT" \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --enable-prefix-caching \
        --host 0.0.0.0 \
        --enforce-eager \
        > "$LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"
    info "Server starting (PID $pid) ..."

    # Wait for server to be ready (up to 120s)
    info "Waiting for server to load model ..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/v1/models" &>/dev/null; then
            info "Server is ready!"
            curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
            return 0
        fi
        # Check if process died
        if ! kill -0 "$pid" 2>/dev/null; then
            error "Server process died. Check $LOG"
        fi
        sleep 2
        printf "."
    done
    echo ""
    warn "Server did not respond within 120s. Check: tail -f $LOG"
}

# --- Stop server ---
do_stop() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping server (PID $pid) ..."
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                warn "Force killing ..."
                kill -9 "$pid"
            fi
            info "Server stopped."
        else
            info "Server not running (stale PID file)."
        fi
        rm -f "$PID_FILE"
    else
        # Try to find by port
        local pid
        pid=$(lsof -ti :"$PORT" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            info "Found process on port $PORT (PID $pid), stopping ..."
            kill "$pid"
            info "Stopped."
        else
            info "No server running."
        fi
    fi
}

# --- Server status ---
do_status() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            info "Server running (PID $pid)"
            if curl -s "http://localhost:$PORT/v1/models" &>/dev/null; then
                info "API responding on port $PORT"
                curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
            else
                warn "Process alive but API not responding yet"
            fi
        else
            warn "PID file exists but process not running"
            rm -f "$PID_FILE"
        fi
    else
        local pid
        pid=$(lsof -ti :"$PORT" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            info "Found server on port $PORT (PID $pid) - no PID file"
        else
            info "No server running."
        fi
    fi

    echo ""
    nvidia-smi 2>/dev/null || true
}

# --- Show logs ---
do_logs() {
    if [ -f "$LOG" ]; then
        tail -f "$LOG"
    else
        error "No log file at $LOG"
    fi
}

# --- Main ---
case "${1:-help}" in
    setup)  do_setup ;;
    start)  do_start ;;
    stop)   do_stop ;;
    restart)
        do_stop
        sleep 2
        do_start
        ;;
    status) do_status ;;
    logs)   do_logs ;;
    *)
        echo "Usage: $0 {setup|start|stop|restart|status|logs}"
        echo ""
        echo "  setup   - Install vLLM, create venv, download deps (first time only)"
        echo "  start   - Start vLLM server (waits until ready)"
        echo "  stop    - Stop vLLM server"
        echo "  restart - Stop then start"
        echo "  status  - Check server status and GPU usage"
        echo "  logs    - Tail server logs"
        ;;
esac
