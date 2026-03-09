#!/bin/bash
# =============================================================================
# BixBench Test: Solve bix-1-q1 using Qwen3.5-27B-AWQ-4bit on local PCIE A100
# sh determines ALL paths/config, python is a pure function.
#
# All artifacts go to: tests/<timestamp>_bixbench/
#   - trace.md         (model streaming output)
#   - bixbench_trace.json (structured trace)
#   - console.log      (terminal output, ANSI stripped)
#   - bixbench_workspace/ (notebook + data)
#
# Usage: bash scripts/PCIE_27B_INT4/step-3-test/test_bixbench.sh
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
SCRIPT_DIR="$WORKDIR/scripts/PCIE_27B_INT4/step-3-test"
TESTS_DIR="$WORKDIR/scripts/PCIE_27B_INT4/tests"
NODE_INFO="$WORKDIR/.node_info_pcie_27b_int4"
DATA_DIR="/pscratch/sd/l/lingzhi/BixBench/bixbench_data"
MAMBA_BIN="/pscratch/sd/l/lingzhi/bin/micromamba"
export MAMBA_ROOT_PREFIX="/pscratch/sd/l/lingzhi/micromamba"

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

# Get API URL
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi
API_URL="${API_URL:-http://localhost:8000/v1}"

# Create timestamped output directory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$TESTS_DIR/${TIMESTAMP}_bixbench"
mkdir -p "$OUTPUT_DIR"

echo -e "${CYAN}=========================================="
echo "  BixBench bix-1-q1 — Qwen3.5-27B-AWQ-4bit"
echo "  vLLM:   $API_URL"
echo "  Output: $OUTPUT_DIR"
echo -e "==========================================${NC}"

# Health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo -e "${RED}ERROR: vLLM not responding at $API_URL${NC}"
    echo "Start it first: bash scripts/PCIE_27B_INT4/step-2-vLLM/serve.sh"
    exit 1
fi

MODEL_NAME=$(curl -s "${API_URL}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Model: $MODEL_NAME"
echo ""

# Call python as a pure function — all config via args
"$MAMBA_BIN" run -n bixbench python3 "$SCRIPT_DIR/test_bixbench.py" \
    --api-url "$API_URL" \
    --model "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    2>&1 | tee >(sed 's/\x1b\[[0-9;]*m//g' > "$OUTPUT_DIR/console.log")

echo ""
echo -e "${GREEN}Output saved to: $OUTPUT_DIR${NC}"
ls -la "$OUTPUT_DIR/"
