#!/usr/bin/env bash
# start-ornith-cpu.sh
#
# Starts Ornith-1.0-9B-GGUF via llama.cpp llama-server on CPU.
# No GPU required. Exposes an OpenAI-compatible API on port 8080.
#
# Prerequisites:
#   - llama.cpp installed (llama-server binary in PATH)
#     Install: https://github.com/ggerganov/llama.cpp#build
#     Quick:   pip install llama-cpp-python[server]
#   - Model downloaded to ORNITH_MODEL_PATH
#     Download: huggingface-cli download \
#       deepreinforce-ai/Ornith-1.0-9B-GGUF \
#       ornith-1.0-9b-q4_k_m.gguf \
#       --local-dir /opt/models/ornith
#
# Usage:
#   ./scripts/start-ornith-cpu.sh
#
# Environment variables (all optional, defaults shown):
#   ORNITH_MODEL_PATH   Path to the GGUF model file
#   ORNITH_PORT         Port to serve on (default: 8080)
#   ORNITH_THREADS      Number of CPU threads (default: nproc)
#   ORNITH_CTX_SIZE     Context window size (default: 8192)
#   ORNITH_BATCH_SIZE   Batch size (default: 512)
#   ORNITH_HOST         Bind host (default: 0.0.0.0)
#
# After starting, set these env vars in ttruthdesk-platform .env:
#   LLM_PROVIDER=ornith_slm
#   ORNITH_SLM_URL=http://localhost:8080
#   ORNITH_SLM_MODEL=ornith-1.0-9b
#
set -euo pipefail

ORNITH_MODEL_PATH="${ORNITH_MODEL_PATH:-/opt/models/ornith/ornith-1.0-9b-q4_k_m.gguf}"
ORNITH_PORT="${ORNITH_PORT:-8080}"
ORNITH_THREADS="${ORNITH_THREADS:-$(nproc)}"
ORNITH_CTX_SIZE="${ORNITH_CTX_SIZE:-8192}"
ORNITH_BATCH_SIZE="${ORNITH_BATCH_SIZE:-512}"
ORNITH_HOST="${ORNITH_HOST:-0.0.0.0}"

# ── Validate prerequisites ─────────────────────────────────────────────────────
if ! command -v llama-server &>/dev/null; then
    # Try llama-cpp-python server as fallback
    if python3 -c "import llama_cpp.server" &>/dev/null 2>&1; then
        echo "[ornith-cpu] Using llama-cpp-python server..."
        exec python3 -m llama_cpp.server \
            --model "${ORNITH_MODEL_PATH}" \
            --host "${ORNITH_HOST}" \
            --port "${ORNITH_PORT}" \
            --n_threads "${ORNITH_THREADS}" \
            --n_ctx "${ORNITH_CTX_SIZE}" \
            --n_batch "${ORNITH_BATCH_SIZE}" \
            --served_model_name "ornith-1.0-9b" \
            --chat_format "chatml"
    fi
    echo "[ornith-cpu] ERROR: llama-server not found in PATH."
    echo "  Install llama.cpp: https://github.com/ggerganov/llama.cpp#build"
    echo "  Or: pip install 'llama-cpp-python[server]'"
    exit 1
fi

if [[ ! -f "${ORNITH_MODEL_PATH}" ]]; then
    echo "[ornith-cpu] ERROR: Model not found at ${ORNITH_MODEL_PATH}"
    echo ""
    echo "  Download the model:"
    echo "    pip install huggingface_hub"
    echo "    huggingface-cli download deepreinforce-ai/Ornith-1.0-9B-GGUF \\"
    echo "      ornith-1.0-9b-q4_k_m.gguf \\"
    echo "      --local-dir \$(dirname ${ORNITH_MODEL_PATH})"
    echo ""
    echo "  Or set ORNITH_MODEL_PATH to point to your GGUF file."
    exit 1
fi

# ── Start llama-server ─────────────────────────────────────────────────────────
echo "[ornith-cpu] Starting Ornith-1.0-9B on CPU"
echo "  Model:    ${ORNITH_MODEL_PATH}"
echo "  Port:     ${ORNITH_PORT}"
echo "  Threads:  ${ORNITH_THREADS}"
echo "  Context:  ${ORNITH_CTX_SIZE} tokens"
echo ""
echo "  OpenAI-compatible API: http://localhost:${ORNITH_PORT}/v1"
echo "  Health check:          http://localhost:${ORNITH_PORT}/health"
echo ""
echo "  To use with ttruthdesk-platform:"
echo "    LLM_PROVIDER=ornith_slm"
echo "    ORNITH_SLM_URL=http://localhost:${ORNITH_PORT}"
echo "    ORNITH_SLM_MODEL=ornith-1.0-9b"
echo ""

exec llama-server \
    --model "${ORNITH_MODEL_PATH}" \
    --host "${ORNITH_HOST}" \
    --port "${ORNITH_PORT}" \
    --threads "${ORNITH_THREADS}" \
    --ctx-size "${ORNITH_CTX_SIZE}" \
    --batch-size "${ORNITH_BATCH_SIZE}" \
    --alias "ornith-1.0-9b" \
    --chat-template "chatml" \
    --log-disable
