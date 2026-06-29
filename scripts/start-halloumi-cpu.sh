#!/usr/bin/env bash
# start-halloumi-cpu.sh
#
# Start HallOumi-8B (oumi-ai/HallOumi-8B) as a local OpenAI-compatible
# HTTP server on port 8001 using llama.cpp (CPU-only, no GPU required).
#
# Prerequisites:
#   - Model downloaded: ./scripts/download-halloumi-model.sh
#   - llama-server installed: pip install 'llama-cpp-python[server]'
#     OR build from source: https://github.com/ggerganov/llama.cpp
#
# Environment variables (all optional):
#   HALLOUMI_MODEL_PATH  — path to the GGUF file
#                          (default: /opt/models/halloumi/halloumi-8b-q4_k_m.gguf)
#   HALLOUMI_PORT        — port to bind (default: 8001)
#   HALLOUMI_THREADS     — CPU threads (default: nproc)
#   HALLOUMI_CTX_SIZE    — context window tokens (default: 4096)
#   HALLOUMI_HOST        — bind host (default: 0.0.0.0)
#
# Usage:
#   ./scripts/start-halloumi-cpu.sh
#
# The server exposes an OpenAI-compatible API at:
#   http://localhost:8001/v1/chat/completions
#
# ttruthdesk-platform integration:
#   Set HALLOUMI_ENABLED=true and HALLOUMI_URL=http://localhost:8001
#   in the .env file to enable HallOumi augmentation for ambiguous verdicts.

set -euo pipefail

MODEL_PATH="${HALLOUMI_MODEL_PATH:-/opt/models/halloumi/halloumi-8b-q4_k_m.gguf}"
PORT="${HALLOUMI_PORT:-8001}"
THREADS="${HALLOUMI_THREADS:-$(nproc)}"
CTX_SIZE="${HALLOUMI_CTX_SIZE:-4096}"
HOST="${HALLOUMI_HOST:-0.0.0.0}"

echo "[halloumi-cpu] Starting HallOumi-8B server"
echo "[halloumi-cpu] Model:    ${MODEL_PATH}"
echo "[halloumi-cpu] Port:     ${PORT}"
echo "[halloumi-cpu] Threads:  ${THREADS}"
echo "[halloumi-cpu] Context:  ${CTX_SIZE} tokens"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "[halloumi-cpu] ERROR: Model file not found at ${MODEL_PATH}"
  echo "[halloumi-cpu] Run ./scripts/download-halloumi-model.sh first."
  exit 1
fi

# ── Try llama-server (llama.cpp binary) ──────────────────────────────────────
if command -v llama-server &>/dev/null; then
  echo "[halloumi-cpu] Using llama-server binary"
  exec llama-server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --threads "${THREADS}" \
    --ctx-size "${CTX_SIZE}" \
    --n-predict 1024 \
    --log-disable
fi

# ── Try llama-cpp-python server ───────────────────────────────────────────────
if python3 -c "import llama_cpp" &>/dev/null; then
  echo "[halloumi-cpu] Using llama-cpp-python server"
  exec python3 -m llama_cpp.server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --n_threads "${THREADS}" \
    --n_ctx "${CTX_SIZE}" \
    --n_predict 1024
fi

# ── Neither available ─────────────────────────────────────────────────────────
echo "[halloumi-cpu] ERROR: Neither llama-server nor llama-cpp-python is installed."
echo "[halloumi-cpu] Install with: pip install 'llama-cpp-python[server]'"
exit 1
