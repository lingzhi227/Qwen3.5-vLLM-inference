#!/bin/bash
# =============================================================================
# Download Qwen3.5-122B-A10B model to local pscratch
# Run on login node (has internet). No GPU needed.
# ~122 GB (BF16) / ~61 GB (FP8 weights auto-quantized at load time)
# Usage: bash scripts/step-0-download/download_Qwen3_5_122B.sh
# =============================================================================
set -euo pipefail

MODEL="Qwen/Qwen3.5-122B-A10B"
LOCAL_MODEL_DIR="/pscratch/sd/l/lingzhi/models/Qwen3.5-122B-A10B"
VLLM_IMAGE="vllm/vllm-openai:latest"

echo "Downloading $MODEL to $LOCAL_MODEL_DIR ..."
echo "This will download ~240 GB. Progress will be shown below."
echo ""

mkdir -p "$LOCAL_MODEL_DIR"

shifter --image="$VLLM_IMAGE" \
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

if [ "$SHARDS" -ge 1 ]; then
    echo "  Status: COMPLETE"
else
    echo "  Status: INCOMPLETE - re-run this script"
fi
