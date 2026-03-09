#!/bin/bash
# =============================================================================
# Stress test the PCIE A100 vLLM server (Qwen3.5-27B-AWQ-4bit)
# sh determines ALL paths/config, python is a pure function.
#
# All artifacts go to: tests/<timestamp>_stress/
#   - console.log        (terminal output, ANSI stripped)
#   - stress_*.json      (structured results)
#   - workspaces/        (per-agent CWL workspaces)
#
# Usage:
#   bash scripts/PCIE_27B_INT4/step-3-test/stress_test.sh           # default: 1 5 10 20 50
#   bash scripts/PCIE_27B_INT4/step-3-test/stress_test.sh 1 2 4 8   # custom ramp
#   bash scripts/PCIE_27B_INT4/step-3-test/stress_test.sh 90        # single level
# =============================================================================
set -euo pipefail

WORKDIR="/pscratch/sd/l/lingzhi/Qwen3.5-vLLM-inference"
SCRIPT_DIR="$WORKDIR/scripts/PCIE_27B_INT4/step-3-test"
TESTS_DIR="$WORKDIR/scripts/PCIE_27B_INT4/tests"
NODE_INFO="$WORKDIR/.node_info_pcie_27b_int4"

# Get API URL from node info
if [ -f "$NODE_INFO" ]; then
    source "$NODE_INFO"
fi
API_URL="${API_URL:-http://localhost:8000/v1}"

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

# Concurrency levels from args
if [ $# -gt 0 ]; then
    LEVELS=("$@")
else
    LEVELS=(1 5 10 20 50)
fi

# Create timestamped output directory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$TESTS_DIR/${TIMESTAMP}_stress"
mkdir -p "$OUTPUT_DIR"

echo -e "${CYAN}=========================================="
echo "  Stress Test — Qwen3.5-27B-AWQ-4bit"
echo "  vLLM:   $API_URL"
echo "  Levels: ${LEVELS[*]}"
echo "  Output: $OUTPUT_DIR"
echo -e "==========================================${NC}"
echo ""

# Health check
if ! curl -s --connect-timeout 5 "${API_URL}/models" > /dev/null; then
    echo -e "${RED}ERROR: Server not responding at $API_URL${NC}"
    echo "Start it first: bash scripts/PCIE_27B_INT4/step-2-vLLM/serve.sh"
    exit 1
fi

MODEL_NAME=$(curl -s "${API_URL}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "Model: $MODEL_NAME"
echo ""

# Locate cwltool
CWLTOOL_BIN=$(python3 -c "import shutil; print(shutil.which('cwltool') or '')")
if [ -z "$CWLTOOL_BIN" ]; then
    CWLTOOL_BIN="$HOME/.local/bin/cwltool"
fi
if [ ! -x "$CWLTOOL_BIN" ]; then
    echo -e "${RED}ERROR: cwltool not found${NC}"
    exit 1
fi
echo "Using cwltool: $CWLTOOL_BIN"
echo ""

# Call python as a pure function — all config via args
python3 "$SCRIPT_DIR/test_cwl_stress.py" \
    --api-url "$API_URL" \
    --model "$MODEL_NAME" \
    --cwltool "$CWLTOOL_BIN" \
    --output-dir "$OUTPUT_DIR" \
    -n "${LEVELS[@]}" \
    2>&1 | tee >(sed 's/\x1b\[[0-9;]*m//g' > "$OUTPUT_DIR/console.log")

echo ""
echo -e "${GREEN}Output saved to: $OUTPUT_DIR${NC}"
ls -la "$OUTPUT_DIR/"
