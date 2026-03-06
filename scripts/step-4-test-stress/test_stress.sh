#!/bin/bash
# =============================================================================
# Stress test: run N concurrent CWL agents against vLLM server
# Usage:
#   bash scripts/test_stress.sh              # default: ramp up 1,3,5,10,15
#   bash scripts/test_stress.sh 5            # test with 5 agents
#   bash scripts/test_stress.sh 1 3 5 10 15  # custom ramp-up levels
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info"

# Get API URL from node info
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

echo "Stress testing vLLM server at: $API_URL"

# Quick health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo "ERROR: Server not responding at $API_URL"
    exit 1
fi

# Install dependencies if needed
python3 -m pip show openai &>/dev/null || python3 -m pip install --user openai
python3 -m pip show cwltool &>/dev/null || python3 -m pip install --user cwltool

# Locate cwltool
CWLTOOL_BIN=$(python3 -c "import shutil; print(shutil.which('cwltool') or '')")
if [ -z "$CWLTOOL_BIN" ]; then
    CWLTOOL_BIN="$HOME/.local/bin/cwltool"
fi

echo "Using cwltool: $CWLTOOL_BIN"

# Set concurrency levels
if [ $# -gt 0 ]; then
    LEVELS="$@"
else
    LEVELS="1 3 5 10 15"
fi

echo "Concurrency levels: $LEVELS"
echo ""

# Run stress test
export VLLM_API_URL="$API_URL"
export VLLM_MODEL="/pscratch/sd/l/lingzhi/models/Qwen3.5-27B"
export CWLTOOL_BIN

python3 "$WORKDIR/tests/coding_cwl/cwl_stress_test.py" -n $LEVELS
