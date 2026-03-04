#!/usr/bin/env bash
# =============================================================================
# One-click deploy: from Mac to remote GPU server
# Usage: bash deploy.sh [setup|start|stop|restart|status|logs|tunnel|demo]
# =============================================================================
set -euo pipefail

# --- Config ---
REMOTE_HOST="108.41.63.249"
REMOTE_PORT=2222
REMOTE_USER="lingzhi"
SSH_KEY="$HOME/.ssh/id_ed25519"
LOCAL_API_PORT=8000
REMOTE_API_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_WORKDIR="qwen-server"

SSH_OPTS="-p $REMOTE_PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

# --- Colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
header(){ echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

# --- SSH helper ---
remote() {
    ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "$@"
}

# --- Upload server.sh to remote ---
upload_script() {
    info "Uploading server.sh to remote ..."
    scp -P "$REMOTE_PORT" -i "$SSH_KEY" \
        "$SCRIPT_DIR/server.sh" \
        "$REMOTE_USER@$REMOTE_HOST:~/$REMOTE_WORKDIR/server.sh"
    remote "chmod +x ~/$REMOTE_WORKDIR/server.sh"
}

# --- First-time setup ---
do_setup() {
    header "First-time Setup"

    # Ensure remote workdir exists
    remote "mkdir -p ~/$REMOTE_WORKDIR"

    # Upload and run setup
    upload_script
    info "Running setup on remote (this may take a few minutes) ..."
    remote "cd ~/$REMOTE_WORKDIR && bash server.sh setup"

    info "Setup complete!"
}

# --- Start server ---
do_start() {
    header "Starting vLLM Server"

    # Ensure latest script is on remote
    remote "mkdir -p ~/$REMOTE_WORKDIR"
    upload_script

    # Start server
    remote "cd ~/$REMOTE_WORKDIR && bash server.sh start"

    # Set up tunnel
    do_tunnel

    info "All done! API available at http://localhost:$LOCAL_API_PORT/v1"
}

# --- Stop server ---
do_stop() {
    header "Stopping vLLM Server"
    remote "cd ~/$REMOTE_WORKDIR && bash server.sh stop"

    # Kill local tunnel
    kill_tunnel
}

# --- Restart ---
do_restart() {
    do_stop
    sleep 2
    do_start
}

# --- Server status ---
do_status() {
    header "Server Status"
    remote "cd ~/$REMOTE_WORKDIR && bash server.sh status"

    echo ""
    # Check local tunnel
    if lsof -i :"$LOCAL_API_PORT" &>/dev/null; then
        info "Local SSH tunnel: active (port $LOCAL_API_PORT)"
        if curl -s "http://localhost:$LOCAL_API_PORT/v1/models" &>/dev/null; then
            info "Local API: responding"
        else
            warn "Local API: tunnel exists but API not responding"
        fi
    else
        warn "Local SSH tunnel: not active"
    fi
}

# --- SSH tunnel management ---
do_tunnel() {
    header "SSH Tunnel"

    # Check if tunnel already exists
    if lsof -i :"$LOCAL_API_PORT" &>/dev/null; then
        if curl -s --connect-timeout 3 "http://localhost:$LOCAL_API_PORT/v1/models" &>/dev/null; then
            info "Tunnel already active and API responding"
            return 0
        fi
        warn "Port $LOCAL_API_PORT in use but API not responding, recreating tunnel ..."
        kill_tunnel
    fi

    info "Creating SSH tunnel (localhost:$LOCAL_API_PORT -> remote:$REMOTE_API_PORT) ..."
    ssh $SSH_OPTS \
        -f -N \
        -L "$LOCAL_API_PORT:localhost:$REMOTE_API_PORT" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        "$REMOTE_USER@$REMOTE_HOST"

    sleep 2

    if curl -s --connect-timeout 5 "http://localhost:$LOCAL_API_PORT/v1/models" &>/dev/null; then
        info "Tunnel created, API responding!"
    else
        warn "Tunnel created but API not yet responding (server may still be loading)"
    fi
}

kill_tunnel() {
    local pids
    pids=$(ps aux | grep "ssh.*-L.*$LOCAL_API_PORT:localhost:$REMOTE_API_PORT" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$pids" ]; then
        info "Killing SSH tunnel (PID: $pids) ..."
        echo "$pids" | xargs kill 2>/dev/null || true
    fi
}

# --- Show remote logs ---
do_logs() {
    header "Server Logs (Ctrl+C to quit)"
    remote "tail -f ~/$REMOTE_WORKDIR/server.log"
}

# --- Run demo ---
do_demo() {
    header "Running Tool Calling Demo"

    # Ensure tunnel is up
    do_tunnel

    cd "$PROJECT_DIR"
    if [ ! -d "venv" ]; then
        info "Creating local venv ..."
        python3 -m venv venv
        source venv/bin/activate
        pip install openai
    else
        source venv/bin/activate
    fi

    python3 tool_demo.py
}

# --- Main ---
case "${1:-help}" in
    setup)   do_setup ;;
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_restart ;;
    status)  do_status ;;
    tunnel)  do_tunnel ;;
    logs)    do_logs ;;
    demo)    do_demo ;;
    *)
        echo -e "${CYAN}Qwen3.5 vLLM Deployment Script${NC}"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  setup   - First-time: install vLLM + deps on remote GPU server"
        echo "  start   - Start vLLM server + SSH tunnel (main command)"
        echo "  stop    - Stop server + tunnel"
        echo "  restart - Stop then start"
        echo "  status  - Check server, GPU, and tunnel status"
        echo "  tunnel  - (Re)create SSH tunnel only"
        echo "  logs    - Tail remote server logs"
        echo "  demo    - Run tool calling demo script"
        echo ""
        echo "Config:"
        echo "  Remote:  $REMOTE_USER@$REMOTE_HOST:$REMOTE_PORT"
        echo "  Model:   Qwen/Qwen3.5-9B (4-bit bitsandbytes)"
        echo "  API:     http://localhost:$LOCAL_API_PORT/v1"
        ;;
esac
