#!/bin/bash
# =============================================================================
# Test the 122B vLLM server with cwl_agent_test.py
# Usage: bash scripts/step-3-test-agentic/test_cwl_agent_122B.sh
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info_122B"
MODEL="/pscratch/sd/l/lingzhi/models/Qwen3.5-122B-A10B"

# Get API URL from node info or SLURM env
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi

if [ -z "${API_URL:-}" ]; then
    if [ -n "${SLURM_NODELIST:-}" ]; then
        API_URL="http://${SLURM_NODELIST}:8000/v1"
    else
        echo "ERROR: No API_URL found. Run step2_serve_122B.sh first."
        exit 1
    fi
fi

echo "Testing 122B vLLM server at: $API_URL"

# Quick health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo "ERROR: Server not responding at $API_URL"
    exit 1
fi

echo "Server is responding. Running CWL agent test ..."
echo ""

# Install dependencies if needed
python3 -m pip show openai &>/dev/null || python3 -m pip install --user openai
python3 -m pip show cwltool &>/dev/null || python3 -m pip install --user cwltool

# Locate cwltool binary
CWLTOOL_BIN=$(python3 -c "import shutil; print(shutil.which('cwltool') or '')")
if [ -z "$CWLTOOL_BIN" ]; then
    CWLTOOL_BIN="$HOME/.local/bin/cwltool"
fi

if [ ! -x "$CWLTOOL_BIN" ]; then
    echo "ERROR: cwltool not found at $CWLTOOL_BIN"
    exit 1
fi

echo "Using cwltool: $CWLTOOL_BIN"
echo ""

# Run CWL agent test
export VLLM_API_URL="$API_URL"
export VLLM_MODEL="$MODEL"
export CWLTOOL_BIN
python3 "$WORKDIR/tests/coding_cwl/cwl_agent_test.py"
