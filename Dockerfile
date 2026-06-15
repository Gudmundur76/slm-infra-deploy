# ── claims-slm fine-tuning container ─────────────────────────────────────────
#
# Multi-stage build:
#   Stage 1 (deps)  — install Python dependencies into a virtual environment
#   Stage 2 (final) — minimal runtime image with Ollama + Python pipeline
#
# Usage:
#   docker build -t claims-slm-trainer .
#   docker run --rm \
#     -v /path/to/corpus:/data/corpus.jsonl:ro \
#     -v /path/to/adapter:/data/adapter \
#     claims-slm-trainer \
#     python finetunePipeline.py --corpus /data/corpus.jsonl --output /data/adapter --cpu

# ── Stage 1: Python dependency builder ───────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /build

# Install build tools for packages that compile C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Create venv and install all Python deps
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Final runtime image ──────────────────────────────────────────────
FROM python:3.11-slim AS final

LABEL org.opencontainers.image.title="claims-slm-trainer"
LABEL org.opencontainers.image.description="LoRA fine-tuning pipeline for the citation.is claims SLM"
LABEL org.opencontainers.image.source="https://github.com/Gudmundur76/slm-infra-deploy"

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (used by IncrementalTrainer to refresh the model after training)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy Python venv from builder stage
COPY --from=deps /opt/venv /opt/venv

# Activate venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application files
COPY finetunePipeline.py .
COPY Modelfile .

# Create data directories
RUN mkdir -p /data/corpus /data/adapter /data/models

# Default corpus and adapter paths (override with -v mounts)
ENV CORPUS_PATH=/data/corpus/corpus.jsonl
ENV ADAPTER_PATH=/data/adapter

# Health check — verify Python and key packages are importable
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import transformers; import peft; print('OK')" || exit 1

# Default command: run the fine-tuning pipeline in CPU mode
CMD ["python", "finetunePipeline.py", \
     "--corpus", "/data/corpus/corpus.jsonl", \
     "--output", "/data/adapter", \
     "--cpu"]
