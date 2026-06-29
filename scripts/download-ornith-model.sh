#!/usr/bin/env bash
# download-ornith-model.sh
#
# Downloads the Ornith-1.0-9B-GGUF Q4_K_M quantisation (~5.5 GB).
# This is the recommended quantisation for CPU inference:
#   - RAM required: ~6.5 GB (model + KV cache)
#   - Throughput: ~3–8 tok/s on a modern 4-core CPU
#   - Quality: near-identical to full bf16 for factual claim verification
#
# Usage:
#   ./scripts/download-ornith-model.sh
#   ./scripts/download-ornith-model.sh /custom/model/dir
#
set -euo pipefail

MODEL_DIR="${1:-/opt/models/ornith}"
REPO="deepreinforce-ai/Ornith-1.0-9B-GGUF"
FILENAME="ornith-1.0-9b-q4_k_m.gguf"

echo "[download-ornith] Downloading ${FILENAME} to ${MODEL_DIR}"
echo "[download-ornith] Source: https://huggingface.co/${REPO}"
echo ""

mkdir -p "${MODEL_DIR}"

# Use huggingface-cli if available, otherwise fall back to wget
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "${REPO}" "${FILENAME}" --local-dir "${MODEL_DIR}"
elif command -v wget &>/dev/null; then
    HF_URL="https://huggingface.co/${REPO}/resolve/main/${FILENAME}"
    wget -c --show-progress -O "${MODEL_DIR}/${FILENAME}" "${HF_URL}"
elif command -v curl &>/dev/null; then
    HF_URL="https://huggingface.co/${REPO}/resolve/main/${FILENAME}"
    curl -L --progress-bar -o "${MODEL_DIR}/${FILENAME}" "${HF_URL}"
else
    echo "[download-ornith] ERROR: Neither huggingface-cli, wget, nor curl found."
    echo "  Install huggingface_hub: pip install huggingface_hub"
    exit 1
fi

echo ""
echo "[download-ornith] Done. Model saved to: ${MODEL_DIR}/${FILENAME}"
echo ""
echo "  Start the server:"
echo "    ORNITH_MODEL_PATH=${MODEL_DIR}/${FILENAME} ./scripts/start-ornith-cpu.sh"
