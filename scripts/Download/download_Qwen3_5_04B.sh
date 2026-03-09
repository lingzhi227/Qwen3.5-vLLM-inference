#!/bin/bash
# =============================================================================
# Download Qwen3.5-0.4B (draft model for speculative decoding)
# Run on login node (has internet). No GPU needed.
# Usage: bash scripts/step-0-download/download_Qwen3_5_04B.sh
# =============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3.5-0.8B"
LOCAL_MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-0.8B"

echo "Downloading $MODEL to $LOCAL_MODEL_DIR ..."
echo "This will download ~1.7 GB."
echo ""

mkdir -p "$LOCAL_MODEL_DIR"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '$MODEL',
    local_dir='$LOCAL_MODEL_DIR',
    ignore_patterns=['*.gguf', '*.onnx'],
)
print()
print('Download complete!')
print('Model saved to: $LOCAL_MODEL_DIR')
"

echo ""
echo "Verifying ..."
SHARDS=$(find "$LOCAL_MODEL_DIR" -name "*.safetensors" | wc -l)
SIZE=$(du -sh "$LOCAL_MODEL_DIR" | cut -f1)
echo "  Shards: $SHARDS"
echo "  Size:   $SIZE"
echo "  Status: COMPLETE"
