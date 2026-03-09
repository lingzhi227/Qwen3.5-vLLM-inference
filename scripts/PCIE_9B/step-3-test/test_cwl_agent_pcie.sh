#!/bin/bash
# =============================================================================
# Test the PCIE A100 vLLM server (Qwen3.5-9B) with cwl_agent_test.py
# Usage: bash scripts/step-3-test-agentic/test_cwl_agent_pcie.sh
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info_pcie"

# Get API URL from PCIE node info
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi

API_URL="${API_URL:-http://localhost:8000/v1}"

echo "Testing vLLM server at: $API_URL"

# Quick health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo "ERROR: Server not responding at $API_URL"
    echo "Start it first: bash scripts/PCIE/serve.sh"
    exit 1
fi

MODEL_NAME=$(curl -s "${API_URL}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "Model: $MODEL_NAME"
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

# Run CWL agent test with 9B model
export VLLM_API_URL="$API_URL"
export VLLM_MODEL="$MODEL_NAME"
export CWLTOOL_BIN
python3 "$WORKDIR/tests/coding_cwl/cwl_agent_test.py"
