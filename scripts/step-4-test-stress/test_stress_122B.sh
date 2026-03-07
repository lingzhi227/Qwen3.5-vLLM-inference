#!/bin/bash
# =============================================================================
# Stress test: run N concurrent CWL agents against 122B vLLM server
# Single vLLM instance with tensor parallelism across 4 GPUs
#
# Usage:
#   bash scripts/step-4-test-stress/test_stress_122B.sh              # default: ramp up 1,3,5,10,15,30
#   bash scripts/step-4-test-stress/test_stress_122B.sh 5             # test with 5 agents
#   bash scripts/step-4-test-stress/test_stress_122B.sh 1 5 10 30 50  # custom ramp-up levels
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Projects/Qwen3.5-vLLM-inference"
NODE_INFO="$WORKDIR/.node_info_122B"
MODEL="/pscratch/sd/l/lingzhi/models/Qwen3.5-122B-A10B"

# Get API URL from node info
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

echo "============================================"
echo "  Stress Test: Qwen3.5-122B-A10B"
echo "  Server: $API_URL"
echo "  Mode: Tensor Parallel (4 GPUs, 1 instance)"
echo "============================================"
echo ""

# Quick health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo "ERROR: Server not responding at $API_URL"
    exit 1
fi

echo "Server is responding."

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
    LEVELS="1 3 5 10 15 30"
fi

echo "Concurrency levels: $LEVELS"
echo ""

# Run stress test
export VLLM_API_URL="$API_URL"
export VLLM_MODEL="$MODEL"
export CWLTOOL_BIN
export STRESS_WORKSPACE_ID="122B"

python3 "$WORKDIR/tests/coding_cwl/cwl_stress_test.py" -n $LEVELS
