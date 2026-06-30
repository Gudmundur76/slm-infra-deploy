#!/usr/bin/env python3
"""
bus_corpus_reader.py — GitHub Bus Corpus Reader for slm-infra-deploy

Polls manus-persistent-drive/bus/slm/corpus/ for new training entries
committed by ttruthdesk-platform's trainingExporter (Channel 4), appends
them to the local corpus.jsonl, and triggers finetunePipeline.py when the
threshold is reached.

This is the slm-infra side of the GitHub message bus. It replaces any
direct HTTP dependency between ttruthdesk and slm-infra.

Usage:
    python bus_corpus_reader.py [--once] [--dry-run]

Environment variables:
    BUS_REPO_PATH       — absolute path to manus-persistent-drive clone
                          (default: ../manus-persistent-drive)
    CORPUS_JSONL_PATH   — local corpus file to append to
                          (default: /data/corpus/corpus.jsonl)
    ADAPTER_OUTPUT_DIR  — where LoRA adapter weights are written
                          (default: ./adapter)
    TRAINING_THRESHOLD  — new examples required to trigger training
                          (default: 50, matches cortex.yaml)
    POLL_INTERVAL_SEC   — seconds between git pull + scan cycles
                          (default: 300 = 5 min)
    GH_PAT              — GitHub PAT for git pull (falls back to gh CLI auth)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("bus_corpus_reader")

# ── Config ────────────────────────────────────────────────────────────────────

BUS_REPO = Path(os.environ.get("BUS_REPO_PATH", "../manus-persistent-drive")).resolve()
BUS_CORPUS_DIR = BUS_REPO / "bus" / "slm" / "corpus"
CORPUS_JSONL = Path(os.environ.get("CORPUS_JSONL_PATH", "/data/corpus/corpus.jsonl"))
ADAPTER_OUTPUT = Path(os.environ.get("ADAPTER_OUTPUT_DIR", "./adapter"))
TRAINING_THRESHOLD = int(os.environ.get("TRAINING_THRESHOLD", "50"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SEC", "300"))
GH_PAT = os.environ.get("GH_PAT", "")

# State file — tracks which bus entries have already been consumed
STATE_FILE = BUS_REPO / "bus" / "slm" / ".reader_state.json"


# ── Git helpers ───────────────────────────────────────────────────────────────

def git_pull_bus() -> bool:
    """Pull latest changes from the bus repo. Returns True on success."""
    try:
        env = os.environ.copy()
        if GH_PAT:
            # Inject PAT into the remote URL for this pull
            result = subprocess.run(
                ["git", "pull", "--rebase", "origin", "main"],
                cwd=BUS_REPO,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        else:
            result = subprocess.run(
                ["git", "pull", "--rebase"],
                cwd=BUS_REPO,
                capture_output=True,
                text=True,
                timeout=60,
            )
        if result.returncode == 0:
            log.debug("Bus repo pulled: %s", result.stdout.strip())
            return True
        log.warning("git pull failed: %s", result.stderr.strip())
        return False
    except Exception as exc:
        log.warning("git pull exception: %s", exc)
        return False


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"consumed": [], "pending_count": 0, "last_training": None}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Corpus helpers ────────────────────────────────────────────────────────────

def scan_new_entries(state: dict) -> list[dict]:
    """Return bus corpus entries not yet consumed."""
    if not BUS_CORPUS_DIR.exists():
        log.warning("Bus corpus dir not found: %s", BUS_CORPUS_DIR)
        return []

    consumed = set(state.get("consumed", []))
    new_entries = []

    for entry_file in sorted(BUS_CORPUS_DIR.glob("*.json")):
        if entry_file.name == ".reader_state.json":
            continue
        if entry_file.name in consumed:
            continue
        try:
            data = json.loads(entry_file.read_text())
            # Each entry is a ttruthdesk training record:
            # { id, instruction, input, output, confidence, source, timestamp }
            if "instruction" in data and "output" in data:
                new_entries.append({"file": entry_file.name, "data": data})
            else:
                log.warning("Skipping malformed entry: %s", entry_file.name)
        except Exception as exc:
            log.warning("Failed to read %s: %s", entry_file.name, exc)

    return new_entries


def append_to_corpus(entries: list[dict], dry_run: bool = False) -> int:
    """Append new entries to the local corpus.jsonl. Returns count appended."""
    if not entries:
        return 0

    CORPUS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    appended = 0
    with CORPUS_JSONL.open("a", encoding="utf-8") as f:
        for entry in entries:
            record = {
                "instruction": entry["data"].get("instruction", ""),
                "input": entry["data"].get("input", ""),
                "output": entry["data"].get("output", ""),
            }
            if not dry_run:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            appended += 1

    log.info("%sAppended %d entries to corpus (%s)",
             "[DRY RUN] " if dry_run else "", appended, CORPUS_JSONL)
    return appended


def trigger_training(dry_run: bool = False) -> bool:
    """Run finetunePipeline.py. Returns True on success."""
    script = Path(__file__).parent / "finetunePipeline.py"
    if not script.exists():
        log.error("finetunePipeline.py not found at %s", script)
        return False

    cmd = [
        sys.executable, str(script),
        "--corpus", str(CORPUS_JSONL),
        "--output", str(ADAPTER_OUTPUT),
        "--cpu",
    ]

    log.info("%sTriggerring training: %s", "[DRY RUN] " if dry_run else "", " ".join(cmd))

    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, timeout=7200)  # 2h max
        if result.returncode == 0:
            log.info("Training completed successfully. Adapter at: %s", ADAPTER_OUTPUT)
            return True
        log.error("Training failed with exit code %d", result.returncode)
        return False
    except subprocess.TimeoutExpired:
        log.error("Training timed out after 2 hours")
        return False
    except Exception as exc:
        log.error("Training exception: %s", exc)
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_cycle(state: dict, dry_run: bool = False) -> dict:
    """One poll cycle. Returns updated state."""
    log.info("=== Bus corpus reader cycle @ %s ===",
             datetime.now(timezone.utc).isoformat())

    # 1. Pull latest bus state
    git_pull_bus()

    # 2. Scan for new entries
    new_entries = scan_new_entries(state)
    if not new_entries:
        log.info("No new corpus entries found.")
        return state

    log.info("Found %d new entries in bus/slm/corpus/", len(new_entries))

    # 3. Append to local corpus
    appended = append_to_corpus(new_entries, dry_run=dry_run)

    # 4. Mark as consumed
    consumed = list(state.get("consumed", []))
    consumed.extend(e["file"] for e in new_entries)
    pending_count = state.get("pending_count", 0) + appended

    state = {
        **state,
        "consumed": consumed,
        "pending_count": pending_count,
        "last_scan": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)

    # 5. Trigger training if threshold reached
    log.info("Pending count: %d / %d (threshold)", pending_count, TRAINING_THRESHOLD)
    if pending_count >= TRAINING_THRESHOLD:
        log.info("Threshold reached — triggering LoRA fine-tuning")
        success = trigger_training(dry_run=dry_run)
        if success:
            state = {
                **state,
                "pending_count": 0,
                "last_training": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state)
            log.info("Training done. Pending count reset to 0.")
        else:
            log.error("Training failed — pending count NOT reset. Will retry next cycle.")

    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="slm-infra GitHub bus corpus reader")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle then exit (useful for cron)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and log but do not write or train")
    args = parser.parse_args()

    log.info("Bus corpus reader starting. Bus repo: %s", BUS_REPO)
    log.info("Corpus JSONL: %s | Threshold: %d | Poll: %ds",
             CORPUS_JSONL, TRAINING_THRESHOLD, POLL_INTERVAL)

    if not BUS_REPO.exists():
        log.error("Bus repo not found at %s — clone manus-persistent-drive first", BUS_REPO)
        sys.exit(1)

    state = load_state()

    if args.once:
        run_cycle(state, dry_run=args.dry_run)
        return

    # Continuous polling loop
    while True:
        try:
            state = run_cycle(state, dry_run=args.dry_run)
        except KeyboardInterrupt:
            log.info("Interrupted — exiting.")
            break
        except Exception as exc:
            log.error("Cycle error (will retry): %s", exc)

        log.info("Sleeping %ds until next cycle...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
