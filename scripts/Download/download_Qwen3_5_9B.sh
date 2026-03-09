#!/bin/bash
# =============================================================================
# Download Qwen3.5-9B model to local storage
# Usage: bash scripts/PCIE/download.sh
# =============================================================================
set -euo pipefail

MODEL_ID="Qwen/Qwen3.5-9B"
MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-9B"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }

echo -e "${CYAN}=========================================="
echo "  Download: $MODEL_ID"
echo "  Target:   $MODEL_DIR"
echo -e "==========================================${NC}"
echo ""

if [ -d "$MODEL_DIR" ] && [ "$(find "$MODEL_DIR" -name '*.safetensors' 2>/dev/null | wc -l)" -gt 0 ]; then
    info "Model already downloaded:"
    du -sh "$MODEL_DIR"
    ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l | xargs -I{} echo "  {} safetensor shards"
    echo ""
    info "To re-download, remove $MODEL_DIR first."
    exit 0
fi

mkdir -p "$MODEL_DIR"

info "Downloading $MODEL_ID from HuggingFace..."
# Use huggingface-cli if available, otherwise Python
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')
print('Download complete')
"
fi

echo ""
info "Download complete:"
du -sh "$MODEL_DIR"
find "$MODEL_DIR" -name "*.safetensors" | wc -l | xargs -I{} echo "  {} safetensor shards"
