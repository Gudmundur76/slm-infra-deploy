#!/usr/bin/env python3
"""
cortex — CLI for cognitive-loop-framework

Usage:
    cortex init --repo ./my-app --domain biotech
    cortex run
    cortex train --corpus /data/corpus.jsonl
    cortex status

This is the developer-facing entry point described in the product spec.
"""

import argparse
import http.server
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── Version ───────────────────────────────────────────────────────────────────
VERSION = "0.2.0"
CORTEX_YAML = "cortex.yaml"

# ── Ornith SLM constants ──────────────────────────────────────────────────────
ORNITH_URL = os.environ.get("ORNITH_SLM_URL", "http://localhost:8080")
ORNITH_MODEL = os.environ.get("ORNITH_SLM_MODEL", "ornith-1.0-9b")
ORNITH_HEALTH_ENDPOINT = f"{ORNITH_URL}/health"
ORNITH_MODELS_ENDPOINT = f"{ORNITH_URL}/v1/models"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command, streaming output unless capture=True."""
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def _require_cortex_yaml() -> Path:
    """Ensure cortex.yaml exists in the current directory."""
    path = Path(CORTEX_YAML)
    if not path.exists():
        print(f"[cortex] Error: {CORTEX_YAML} not found in current directory.")
        print("         Run `cortex init` first.")
        sys.exit(1)
    return path


def _print_step(step: int, total: int, msg: str) -> None:
    print(f"[cortex] ({step}/{total}) {msg}")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> None:
    """
    cortex init --repo <path> --domain <domain>

    Initialises the cognitive loop for a repository:
    1. Copies cortex.yaml template to the target directory
    2. Creates data directories
    3. Pulls the base Ollama model
    4. Creates the claims-slm model from the Modelfile
    """
    repo = Path(args.repo).resolve()
    domain = args.domain
    total_steps = 6

    print(f"\n[cortex] Initialising cognitive loop for: {repo}")
    print(f"[cortex] Domain: {domain}\n")

    # Step 1: Validate repo directory
    _print_step(1, total_steps, f"Validating repository at {repo}")
    if not repo.exists():
        print(f"[cortex] Error: Repository directory not found: {repo}")
        sys.exit(1)

    # Step 2: Write cortex.yaml
    _print_step(2, total_steps, f"Writing {CORTEX_YAML}")
    cortex_yaml_src = Path(__file__).parent / CORTEX_YAML
    cortex_yaml_dst = repo / CORTEX_YAML
    if cortex_yaml_dst.exists() and not args.force:
        print(f"[cortex] {CORTEX_YAML} already exists. Use --force to overwrite.")
    else:
        import shutil
        shutil.copy2(str(cortex_yaml_src), str(cortex_yaml_dst))
        # Patch domain
        content = cortex_yaml_dst.read_text()
        content = content.replace("domain: biotech", f"domain: {domain}")
        content = content.replace("project: citation-is", f"project: {repo.name}")
        cortex_yaml_dst.write_text(content)
        print(f"[cortex] Written: {cortex_yaml_dst}")

    # Step 3: Create data directories
    _print_step(3, total_steps, "Creating data directories")
    for d in ["data/corpus", "data/adapter", "data/corpus/backups"]:
        (repo / d).mkdir(parents=True, exist_ok=True)
    print(f"[cortex] Directories: {repo}/data/{{corpus,adapter}}")

    # Step 4: Pull base Ollama model
    _print_step(4, total_steps, "Pulling base model: qwen2.5-coder:1.5b-instruct")
    try:
        _run(["ollama", "pull", "qwen2.5-coder:1.5b-instruct"])
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[cortex] Warning: Could not pull Ollama model: {exc}")
        print("[cortex] Ensure Ollama is running: https://ollama.com")

    # Step 5: Create claims-slm model
    _print_step(5, total_steps, "Creating claims-slm model from Modelfile")
    modelfile = Path(__file__).parent / "Modelfile"
    try:
        _run(["ollama", "create", "claims-slm", "-f", str(modelfile)])
        print("[cortex] Model created: claims-slm")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[cortex] Warning: Could not create Ollama model: {exc}")

    # Step 6: Check Ornith SLM availability (optional — GPU required)
    _print_step(6, total_steps, f"Checking Ornith SLM at {ORNITH_URL}")
    try:
        import urllib.request as _req
        import urllib.error as _err
        with _req.urlopen(ORNITH_HEALTH_ENDPOINT, timeout=3) as _resp:
            print(f"[cortex] Ornith SLM: running at {ORNITH_URL}")
            print(f"[cortex] Set LLM_PROVIDER=ornith_slm in ttruthdesk .env to use it.")
    except (_err.URLError, OSError):
        print(f"[cortex] Ornith SLM: not running (optional — requires GPU)")
        print(f"[cortex] To start: docker compose --profile ornith up -d ornith-vllm")

    print(f"\n[cortex] Initialisation complete.")
    print(f"[cortex] Next steps:")
    print(f"  1. Edit {cortex_yaml_dst} to configure your domain sources and rules")
    print(f"  2. Run `cortex run` to start the cognitive loop")
    print(f"  3. Run `cortex run --ornith` to also start the Ornith SLM (GPU required)")
    print(f"  4. POST claims to http://localhost:3100/cognitive/ingest\n")


def cmd_run(args: argparse.Namespace) -> None:
    """
    cortex run [--ornith]

    Starts the cognitive loop stack via docker compose.
    Pass --ornith to also start the Ornith SLM vLLM service (requires GPU).
    """
    _require_cortex_yaml()
    compose_file = Path(__file__).parent / "docker-compose.yml"

    services = ["ollama", "cognitive-loop"]
    if getattr(args, "ornith", False):
        services.append("ornith-vllm")
        print("[cortex] Starting cognitive loop stack (with Ornith SLM)...")
    else:
        print("[cortex] Starting cognitive loop stack...")

    try:
        cmd_args = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
        if getattr(args, "ornith", False):
            cmd_args += ["--profile", "ornith"]
        cmd_args += services
        _run(cmd_args)
        print("[cortex] Stack started.")
        print("[cortex] Cognitive loop API: http://localhost:3100")
        print("[cortex] Ollama API:          http://localhost:11434")
        if getattr(args, "ornith", False):
            print(f"[cortex] Ornith SLM API:     {ORNITH_URL}/v1")
            print(f"[cortex] Set LLM_PROVIDER=ornith_slm in ttruthdesk .env")
    except subprocess.CalledProcessError as exc:
        print(f"[cortex] Error starting stack: {exc}")
        sys.exit(1)


def cmd_train(args: argparse.Namespace) -> None:
    """
    cortex train --corpus <path>

    Runs the LoRA fine-tuning pipeline on the given corpus.
    After training, refreshes the Ollama model.
    """
    corpus = Path(args.corpus).resolve()
    if not corpus.exists():
        print(f"[cortex] Error: Corpus file not found: {corpus}")
        sys.exit(1)

    pipeline = Path(__file__).parent / "finetunePipeline.py"
    adapter_dir = Path(args.output).resolve()
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cortex] Starting fine-tuning pipeline...")
    print(f"[cortex] Corpus:  {corpus}")
    print(f"[cortex] Adapter: {adapter_dir}")

    cmd = [
        sys.executable,
        str(pipeline),
        "--corpus", str(corpus),
        "--output", str(adapter_dir),
        "--cpu",
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"[cortex] Training failed: {exc}")
        sys.exit(2)

    # Refresh Ollama model
    modelfile = Path(__file__).parent / "Modelfile"
    print("[cortex] Refreshing claims-slm model...")
    try:
        _run(["ollama", "create", "claims-slm", "-f", str(modelfile)])
        print("[cortex] Model refreshed: claims-slm")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[cortex] Warning: Could not refresh Ollama model: {exc}")

    # If Ornith SLM is running, notify that it picks up adapter weights via vLLM LoRA
    if getattr(args, "ornith", False):
        print("[cortex] Checking Ornith SLM for LoRA adapter hot-reload...")
        try:
            import urllib.request as _req
            import urllib.error as _err
            with _req.urlopen(ORNITH_HEALTH_ENDPOINT, timeout=3):
                print(f"[cortex] Ornith SLM is running at {ORNITH_URL}.")
                print(f"[cortex] Adapter weights written to: {adapter_dir}")
                print(f"[cortex] Restart ornith-vllm to load updated adapter:")
                print(f"[cortex]   docker compose restart ornith-vllm")
        except (_err.URLError, OSError):
            print(f"[cortex] Ornith SLM not running — start with: cortex run --ornith")

    print("[cortex] Training complete.")


def cmd_status(args: argparse.Namespace) -> None:
    """
    cortex status

    Shows the status of the cognitive loop stack and the Ollama model.
    """
    import urllib.request
    import urllib.error

    print("[cortex] Checking cognitive loop status...\n")

    # Check Ollama
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            claims_slm_present = any("claims-slm" in m for m in models)
            print(f"  Ollama:      running")
            print(f"  claims-slm:  {'present' if claims_slm_present else 'NOT FOUND (run: cortex init)'}")
            print(f"  All models:  {', '.join(models) or 'none'}")
    except (urllib.error.URLError, OSError):
        print("  Ollama:      NOT RUNNING (start with: ollama serve)")

    # Check cognitive loop API
    try:
        with urllib.request.urlopen("http://localhost:3100/health", timeout=3) as resp:
            data = json.loads(resp.read())
            print(f"  Cognitive loop API: running — {data}")
    except (urllib.error.URLError, OSError):
        print("  Cognitive loop API: NOT RUNNING (start with: cortex run)")

    # Check Ornith SLM
    try:
        with urllib.request.urlopen(ORNITH_HEALTH_ENDPOINT, timeout=3):
            # Also list loaded models
            try:
                with urllib.request.urlopen(ORNITH_MODELS_ENDPOINT, timeout=3) as resp2:
                    mdata = json.loads(resp2.read())
                    ornith_models = [m.get("id", "?") for m in mdata.get("data", [])]
                    print(f"  Ornith SLM:  running at {ORNITH_URL}")
                    print(f"  Ornith models: {', '.join(ornith_models) or 'none'}")
            except Exception:
                print(f"  Ornith SLM:  running at {ORNITH_URL} (model list unavailable)")
    except (urllib.error.URLError, OSError):
        print(f"  Ornith SLM:  NOT RUNNING (start with: cortex run --ornith)")

    print()


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"cortex v{VERSION} — cognitive-loop-framework CLI")


# ── HTTP /verify endpoint ─────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("CLAIM_VERIFIER_MODEL", "claim-verifier")
SERVE_PORT = int(os.environ.get("CORTEX_SERVE_PORT", "8765"))
# LLM_PROVIDER selects the inference backend for POST /verify.
# "ollama"      — Ollama /api/generate (default, CPU-friendly)
# "ornith_slm"  — Ornith-1.0 OpenAI-compatible endpoint (llama-server or vLLM)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")


def _ollama_generate(claim: str) -> dict:
    """Call Ollama /api/generate and return parsed JSON verdict."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "prompt": claim,
        "stream": False,
        "format": "json",
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return json.loads(body.get("response", "{}"))


class _VerifyHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler for POST /verify."""

    def log_message(self, fmt: str, *args: object) -> None:  # silence default logs
        pass

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/verify":
            self.send_error(404, "Not Found")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        claim = body.get("claim", "")
        if not claim:
            self._json(400, {"error": "Missing 'claim' field"})
            return
        try:
            result = _generate(claim)
            self._json(200, result)
        except Exception as exc:
            self._json(503, {"error": "Local model unavailable", "fallback": True, "detail": str(exc)})

    def _json(self, status: int, data: dict) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _ornith_generate(claim: str) -> dict:
    """
    Call the Ornith-1.0 OpenAI-compatible /v1/chat/completions endpoint
    and return a parsed JSON verdict dict.

    Works with both:
      - llama.cpp llama-server (CPU, no GPU required)
      - vLLM ornith-vllm service (GPU)
    Both expose the same OpenAI-compatible API on ORNITH_URL.

    Ornith returns a <think>...</think> reasoning block before the JSON
    answer. We strip the reasoning block and parse the JSON payload.
    """
    import re as _re
    ornith_base = ORNITH_URL.rstrip("/")
    system_prompt = (
        "You are a scientific claim verification engine. "
        "Given a claim, return a JSON object with: "
        "verdict (one of Supported, Contradicted, Partially Supported, "
        "Ambiguous, Insufficient Evidence, Out of Scope, Needs Expert Review), "
        "confidence (0.0\u20131.0), reasoning (brief explanation), "
        "sources (array of {database, id, url}). "
        "Respond only with valid JSON. No markdown."
    )
    payload = json.dumps({
        "model": ORNITH_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": claim},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{ornith_base}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('ORNITH_SLM_API_KEY', 'ornith-local')}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    content: str = body["choices"][0]["message"]["content"]
    # Strip Ornith <think>...</think> reasoning block
    content = _re.sub(r"<think>[\s\S]*?</think>\s*", "", content).strip()
    # Handle markdown code fences if model wraps output
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


def _generate(claim: str) -> dict:
    """
    Route claim verification to the configured LLM backend.
    LLM_PROVIDER=ollama      → Ollama /api/generate (default)
    LLM_PROVIDER=ornith_slm  → Ornith-1.0 OpenAI-compatible endpoint
    """
    if LLM_PROVIDER == "ornith_slm":
        return _ornith_generate(claim)
    return _ollama_generate(claim)


def cmd_serve(_args: argparse.Namespace) -> None:
    """Start the /verify HTTP server on CORTEX_SERVE_PORT (default 8765)."""
    print(f"[cortex] Starting /verify server on port {SERVE_PORT} (model: {MODEL_NAME})")
    server = http.server.HTTPServer(("", SERVE_PORT), _VerifyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[cortex] Server stopped.")


# ── Argument parser ───────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="cognitive-loop-framework CLI — Drop your codebase. Deploy a self-improving agent.",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Initialise cognitive loop for a repository")
    p_init.add_argument("--repo", default=".", help="Path to the repository (default: .)")
    p_init.add_argument("--domain", default="general",
                        choices=["biotech", "legal", "finance", "general"],
                        help="Domain vertical (default: general)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing cortex.yaml")

    # run
    p_run = sub.add_parser("run", help="Start the cognitive loop stack")
    p_run.add_argument(
        "--ornith", action="store_true",
        help="Also start the Ornith SLM vLLM service (requires NVIDIA GPU)",
    )

    # train
    p_train = sub.add_parser("train", help="Run the LoRA fine-tuning pipeline")
    p_train.add_argument("--corpus", required=True, help="Path to JSONL training corpus")
    p_train.add_argument("--output", default="./adapter", help="Output directory for adapter weights")
    p_train.add_argument(
        "--ornith", action="store_true",
        help="After training, notify Ornith SLM to reload adapter weights",
    )

    # status
    sub.add_parser("status", help="Show cognitive loop stack status")
    # serve
    sub.add_parser("serve", help="Start the /verify HTTP server (port CORTEX_SERVE_PORT, default 8765)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.version or not args.command:
        cmd_version(args)
        if not args.command:
            parser.print_help()
        return

    dispatch = {
        "init": cmd_init,
        "run": cmd_run,
        "train": cmd_train,
        "status": cmd_status,
        "serve": cmd_serve,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
