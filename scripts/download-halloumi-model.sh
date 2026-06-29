#!/usr/bin/env bash
# download-halloumi-model.sh
#
# Download the HallOumi-8B GGUF model (Q4_K_M quantisation, ~5.5 GB).
# Source: https://huggingface.co/oumi-ai/HallOumi-8B (GGUF quantisations)
#
# Usage:
#   ./scripts/download-halloumi-model.sh [destination_dir]
#   Default destination: /opt/models/halloumi/

set -euo pipefail

DEST_DIR="${1:-/opt/models/halloumi}"
MODEL_FILE="halloumi-8b-q4_k_m.gguf"
# HuggingFace GGUF repo for HallOumi-8B (community quantisation)
HF_REPO="bartowski/oumi-ai_HallOumi-8B-GGUF"
MODEL_URL="https://huggingface.co/${HF_REPO}/resolve/main/${MODEL_FILE}"

mkdir -p "${DEST_DIR}"
DEST_FILE="${DEST_DIR}/${MODEL_FILE}"

if [ -f "${DEST_FILE}" ]; then
  echo "[download-halloumi] Model already exists at ${DEST_FILE}"
  echo "[download-halloumi] Delete it and re-run to force re-download."
  exit 0
fi

echo "[download-halloumi] Downloading HallOumi-8B Q4_K_M (~5.5 GB)..."
echo "[download-halloumi] Source: ${MODEL_URL}"
echo "[download-halloumi] Destination: ${DEST_FILE}"

# ── Try huggingface-cli (resumable) ──────────────────────────────────────────
if command -v huggingface-cli &>/dev/null; then
  echo "[download-halloumi] Using huggingface-cli (resumable download)"
  huggingface-cli download "${HF_REPO}" "${MODEL_FILE}" \
    --local-dir "${DEST_DIR}" \
    --local-dir-use-symlinks False
  echo "[download-halloumi] Done: ${DEST_FILE}"
  exit 0
fi

# ── Try wget ──────────────────────────────────────────────────────────────────
if command -v wget &>/dev/null; then
  echo "[download-halloumi] Using wget"
  wget -c -O "${DEST_FILE}" "${MODEL_URL}"
  echo "[download-halloumi] Done: ${DEST_FILE}"
  exit 0
fi

# ── Try curl ─────────────────────────────────────────────────────────────────
if command -v curl &>/dev/null; then
  echo "[download-halloumi] Using curl"
  curl -L -C - -o "${DEST_FILE}" "${MODEL_URL}"
  echo "[download-halloumi] Done: ${DEST_FILE}"
  exit 0
fi

echo "[download-halloumi] ERROR: No download tool found."
echo "[download-halloumi] Install one of: huggingface-cli, wget, curl"
exit 1
