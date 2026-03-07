#!/bin/bash
# =============================================================================
# Test the vLLM server with tool_demo.py
# Usage: bash scripts/step-3-test-agentic/test_tool_demo.sh
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info"

# Get API URL from node info or SLURM env
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi

if [ -z "${API_URL:-}" ]; then
    if [ -n "${SLURM_NODELIST:-}" ]; then
        API_URL="http://${SLURM_NODELIST}:8000/v1"
    else
        echo "ERROR: No API_URL found. Run step1_alloc.sh and step2_serve.sh first."
        exit 1
    fi
fi

echo "Testing vLLM server at: $API_URL"

# Quick health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo "ERROR: Server not responding at $API_URL"
    exit 1
fi

echo "Server is responding. Running tool demo ..."
echo ""

# Install openai if needed
python3 -m pip show openai &>/dev/null || python3 -m pip install --user openai

# Run tool demo with env overrides
export VLLM_API_URL="$API_URL"
export VLLM_MODEL="/pscratch/sd/l/lingzhi/models/Qwen3.5-27B"
python3 "$WORKDIR/tests/tool/tool_demo.py"
