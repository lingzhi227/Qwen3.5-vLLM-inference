#!/bin/bash
# =============================================================================
# Download Qwen3.5-27B-AWQ-4bit model to local storage
# ~20 GB (hybrid arch: DeltaNet layers also quantized to 4-bit)
# Fits on A100-PCIE-40GB with 256K context (20GB model + 16GB KV + 2GB overhead)
#
# Note: Official Qwen/Qwen3.5-27B-GPTQ-Int4 is 30.3GB (DeltaNet layers at 16-bit)
#       which leaves too little room for 256K KV cache on 40GB GPU.
#       This community AWQ-4bit version is smaller at 20GB.
#
# Usage: bash scripts/PCIE_27B_INT4/download.sh
# =============================================================================
set -euo pipefail

MODEL_ID="cyankiwi/Qwen3.5-27B-AWQ-4bit"
MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-27B-AWQ-4bit"
VLLM_IMAGE="vllm/vllm-openai:qwen3_5"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }

echo -e "${CYAN}=========================================="
echo "  Download: $MODEL_ID (AWQ 4-bit)"
echo "  Target:   $MODEL_DIR"
echo "  Expected: ~20 GB"
echo -e "==========================================${NC}"
echo ""

if [ -d "$MODEL_DIR" ] && [ "$(find "$MODEL_DIR" -name '*.safetensors' 2>/dev/null | wc -l)" -gt 0 ]; then
    info "Model already downloaded:"
    du -sh "$MODEL_DIR"
    find "$MODEL_DIR" -name "*.safetensors" | wc -l | xargs -I{} echo "  {} safetensor shards"
    echo ""
    info "To re-download, remove $MODEL_DIR first."
    exit 0
fi

mkdir -p "$MODEL_DIR"

info "Downloading $MODEL_ID from HuggingFace..."
# Try huggingface-cli first, then shifter, then plain python
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
else
    shifter --image="$VLLM_IMAGE" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '$MODEL_ID',
    local_dir='$MODEL_DIR',
    ignore_patterns=['*.gguf', '*.onnx'],
)
print()
print('Download complete!')
"
fi

echo ""
info "Download complete:"
du -sh "$MODEL_DIR"
find "$MODEL_DIR" -name "*.safetensors" | wc -l | xargs -I{} echo "  {} safetensor shards"
