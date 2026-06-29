#!/usr/bin/env python3
"""
oumiPipeline.py — Oumi-based incremental LoRA fine-tuning for claims-slm.

Replaces finetunePipeline.py with a cleaner Oumi CLI-driven workflow.
Called by IncrementalTrainer (cognitive-loop-framework) as:
    python oumiPipeline.py --corpus <path> --output <path> [--cpu]

Input:  JSONL file — each line: {"instruction": str, "input": str, "output": str}
Output: LoRA adapter weights directory at --output path.

Design:
  - Base model: Qwen/Qwen2.5-1.5B-Instruct (CPU-feasible, ~3 GB)
  - Fine-tuning: Oumi SFT + LoRA via configs/oumi_cpu_train.yaml
  - Converts Alpaca-format JSONL → HuggingFace dataset in /tmp
  - Calls `oumi train` as a subprocess (resumable, cloud-launchable)
  - Falls back to finetunePipeline.py if oumi is not installed

Exit codes:
  0  — success, adapter weights written to --output
  1  — corpus file not found or empty
  2  — training error
  3  — oumi not installed and fallback also failed
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("oumiPipeline")

# ── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CONFIG_TEMPLATE = SCRIPT_DIR / "configs" / "oumi_cpu_train.yaml"
ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
MIN_CORPUS_SIZE = 10


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oumi-based claims-slm fine-tuning")
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus file")
    parser.add_argument("--output", required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (default: auto)")
    parser.add_argument(
        "--config",
        default=str(CONFIG_TEMPLATE),
        help="Path to Oumi YAML training config",
    )
    return parser.parse_args()


# ── Corpus loading ────────────────────────────────────────────────────────────
def load_corpus(corpus_path: str) -> list[dict]:
    """Load and validate the JSONL corpus. Raises SystemExit on failure."""
    path = Path(corpus_path)
    if not path.exists():
        log.error("Corpus file not found: %s", corpus_path)
        sys.exit(1)

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if all(k in rec for k in ("instruction", "input", "output")):
                    records.append(rec)
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSON on line %d", lineno)

    if len(records) < MIN_CORPUS_SIZE:
        log.error(
            "Corpus too small: %d valid records (minimum %d required).",
            len(records),
            MIN_CORPUS_SIZE,
        )
        sys.exit(1)

    log.info("Loaded %d training examples from %s", len(records), corpus_path)
    return records


# ── Dataset conversion ────────────────────────────────────────────────────────
def convert_to_oumi_dataset(records: list[dict], dataset_dir: Path) -> None:
    """
    Convert Alpaca-format records to a HuggingFace dataset directory
    that Oumi's PromptResponseDataset can load.

    Each record becomes a JSON file with a 'messages' field in chat format:
      [{"role": "user", "content": <prompt>}, {"role": "assistant", "content": <response>}]
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        prompt = ALPACA_TEMPLATE.format(
            instruction=rec["instruction"],
            input=rec.get("input", ""),
            output="",
        ).rstrip()
        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rec["output"]},
            ]
        })

    # Write as a JSONL file that Oumi's HuggingFaceDataset can load from disk
    jsonl_path = dataset_dir / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Write a minimal dataset_info.json so HF datasets can load it
    info = {
        "dataset_name": "claims_corpus",
        "features": {
            "messages": {
                "dtype": "string",
                "_type": "Value",
            }
        },
        "splits": {"train": {"name": "train", "num_examples": len(rows)}},
    }
    with open(dataset_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    log.info("Wrote %d examples to %s", len(rows), jsonl_path)


# ── Config patching ───────────────────────────────────────────────────────────
def build_oumi_config(
    template_path: str,
    dataset_path: str,
    output_dir: str,
    tmp_dir: Path,
) -> Path:
    """
    Read the YAML template, substitute __CORPUS_DATASET_PATH__ and
    __OUTPUT_DIR__ placeholders, and write a resolved config to tmp_dir.
    """
    with open(template_path, encoding="utf-8") as f:
        content = f.read()

    content = content.replace("__CORPUS_DATASET_PATH__", dataset_path)
    content = content.replace("__OUTPUT_DIR__", output_dir)

    resolved_path = tmp_dir / "oumi_train_resolved.yaml"
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(content)

    return resolved_path


# ── Oumi training ─────────────────────────────────────────────────────────────
def run_oumi_train(config_path: Path) -> None:
    """Run `oumi train` as a subprocess. Raises RuntimeError on failure."""
    cmd = ["oumi", "train", "-c", str(config_path)]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"oumi train exited with code {result.returncode}")


# ── Fallback ──────────────────────────────────────────────────────────────────
def run_fallback_pipeline(corpus: str, output: str, cpu: bool) -> None:
    """Fall back to finetunePipeline.py if oumi is not installed."""
    fallback = SCRIPT_DIR / "finetunePipeline.py"
    if not fallback.exists():
        log.error("Neither oumi nor finetunePipeline.py is available.")
        sys.exit(3)

    cmd = [sys.executable, str(fallback), "--corpus", corpus, "--output", output]
    if cpu:
        cmd.append("--cpu")
    log.info("oumi not found — falling back to finetunePipeline.py")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        log.error("Fallback pipeline failed with exit code %d", result.returncode)
        sys.exit(result.returncode)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Check if oumi is installed
    if shutil.which("oumi") is None:
        log.warning("oumi CLI not found — running fallback pipeline")
        run_fallback_pipeline(args.corpus, args.output, args.cpu)
        return

    records = load_corpus(args.corpus)

    with tempfile.TemporaryDirectory(prefix="oumi_claims_") as tmp:
        tmp_dir = Path(tmp)
        dataset_dir = tmp_dir / "dataset"
        convert_to_oumi_dataset(records, dataset_dir)

        config_path = build_oumi_config(
            template_path=args.config,
            dataset_path=str(dataset_dir),
            output_dir=args.output,
            tmp_dir=tmp_dir,
        )

        try:
            run_oumi_train(config_path)
        except RuntimeError as e:
            log.error("Oumi training failed: %s", e)
            sys.exit(2)

    log.info("Training complete. Adapter written to: %s", args.output)


if __name__ == "__main__":
    main()
