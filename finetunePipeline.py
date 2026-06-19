#!/usr/bin/env python3
"""
finetunePipeline.py — Incremental LoRA fine-tuning for claims-slm.

Called by IncrementalTrainer (cognitive-loop-framework) as:
    python finetunePipeline.py --corpus <path> --output <path> --cpu

Input:  JSONL file — each line: {"instruction": str, "input": str, "output": str}
Output: LoRA adapter weights directory at --output path.

Design:
  - Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct (CPU-feasible, ~3 GB)
  - Fine-tuning: LoRA via peft (r=8, alpha=16, q_proj + v_proj)
  - Training: 3 epochs, batch_size=1, gradient_accumulation=4
  - Incremental: only trains on examples in the provided corpus file
  - CPU-only: no CUDA required; uses float32

Exit codes:
  0  — success, adapter weights written to --output
  1  — corpus file not found or empty
  2  — training error
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("finetunePipeline")

# ── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.05
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LEN = 512
LEARNING_RATE = 2e-4


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental LoRA fine-tuning for claims-slm"
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to JSONL training corpus file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write LoRA adapter weights",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU-only training (no CUDA)",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL_ID,
        help=f"HuggingFace model ID for base model (default: {BASE_MODEL_ID})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Maximum sequence length (default: {MAX_SEQ_LEN})",
    )
    return parser.parse_args()


# ── Corpus loading ────────────────────────────────────────────────────────────
def load_corpus(corpus_path: str) -> list[dict]:
    """Load and validate the JSONL training corpus."""
    path = Path(corpus_path)
    if not path.exists():
        log.error("Corpus file not found: %s", corpus_path)
        sys.exit(1)

    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSON on line %d: %s", line_num, exc)
                continue

            # Validate required fields
            if not all(k in record for k in ("instruction", "input", "output")):
                log.warning(
                    "Skipping line %d: missing instruction/input/output fields",
                    line_num,
                )
                continue
            records.append(record)

        if not records:
            log.error("Corpus is empty after parsing: %s", corpus_path)
            sys.exit(1)
    if not records:
        log.error("Corpus is empty after parsing: %s", corpus_path)
        sys.exit(1)
    if len(records) < 10:
        log.error(
            "Corpus has only %d examples (minimum 10 required): %s",
            len(records),
            corpus_path,
        )
        sys.exit(1)
    log.info("Loaded %d training examples from %s", len(records), corpus_path)
    return records


# ── Prompt formatting ─────────────────────────────────────────────────────────
def format_prompt(record: dict) -> str:
    """Format a training record into an Alpaca-style prompt."""
    return ALPACA_TEMPLATE.format(
        instruction=record["instruction"],
        input=record.get("input", ""),
        output=record["output"],
    )


# ── Dataset construction ──────────────────────────────────────────────────────
def build_hf_dataset(records: list[dict], tokenizer, max_seq_len: int):
    """Build a HuggingFace Dataset from the training records."""
    # Import here so the module can be imported without heavy deps for tests
    from datasets import Dataset  # type: ignore

    texts = [format_prompt(r) for r in records]

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        # For causal LM: labels = input_ids (shifted internally by the model)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    raw_ds = Dataset.from_dict({"text": texts})
    tokenized_ds = raw_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_ds.set_format("torch")
    return tokenized_ds


# ── Model + LoRA setup ────────────────────────────────────────────────────────
def load_model_and_tokenizer(base_model_id: str, cpu_mode: bool):
    """Load the base model and tokenizer, apply LoRA config."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    device_map = "cpu" if cpu_mode else "auto"
    torch_dtype = torch.float32 if cpu_mode else torch.float16

    log.info("Loading tokenizer from %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base model from %s (device_map=%s)", base_model_id, device_map)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )

    log.info(
        "Applying LoRA config: r=%d, alpha=%d, modules=%s",
        LORA_R,
        LORA_ALPHA,
        LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────
def train(
    model,
    tokenizer,
    dataset,
    output_path: str,
    num_epochs: int,
    cpu_mode: bool,
) -> None:
    """Run the training loop and save adapter weights."""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling  # type: ignore

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory inside output
    checkpoint_dir = str(output_dir / "checkpoints")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",          # Don't save HF checkpoints — we save adapter only
        fp16=False,                  # CPU: no fp16
        bf16=False,
        dataloader_num_workers=0,    # CPU: no multiprocessing overhead
        report_to="none",            # No wandb / tensorboard
        no_cuda=cpu_mode,
        use_cpu=cpu_mode,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    log.info(
        "Starting training: %d examples, %d epochs, batch=%d, grad_accum=%d",
        len(dataset),
        num_epochs,
        BATCH_SIZE,
        GRAD_ACCUM,
    )
    trainer.train()
    log.info("Training complete. Saving LoRA adapter to %s", output_path)

    # Save only the LoRA adapter weights (not the full model)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log.info("Adapter weights saved: %s", list(output_dir.iterdir()))


# ── Delta tracking ────────────────────────────────────────────────────────────
def get_trained_count_path(output_path: str) -> Path:
    """Return path to the file that tracks how many examples were trained."""
    return Path(output_path) / ".trained_count"


def read_trained_count(output_path: str) -> int:
    """Read the number of examples trained in the last run."""
    count_path = get_trained_count_path(output_path)
    if not count_path.exists():
        return 0
    try:
        return int(count_path.read_text().strip())
    except (ValueError, OSError):
        return 0


def write_trained_count(output_path: str, count: int) -> None:
    """Write the number of examples trained so far."""
    count_path = get_trained_count_path(output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    count_path.write_text(str(count))


def get_delta_records(
    records: list[dict], output_path: str
) -> tuple[list[dict], int]:
    """
    Return only the new records since the last training run.
    This implements incremental (delta) training — we don't retrain on
    examples the model has already seen.
    """
    trained_count = read_trained_count(output_path)
    total = len(records)

    if trained_count >= total:
        log.info(
            "No new examples since last run (trained=%d, total=%d). Nothing to do.",
            trained_count,
            total,
        )
        return [], trained_count

    delta = records[trained_count:]
    log.info(
        "Delta training: %d new examples (trained=%d → total=%d)",
        len(delta),
        trained_count,
        total,
    )
    return delta, total


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.cpu:
        # Ensure CUDA is not used even if available
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log.info("CPU mode enabled — CUDA disabled")

    # 1. Load corpus
    records = load_corpus(args.corpus)

    # 2. Determine delta (incremental training)
    delta_records, new_total = get_delta_records(records, args.output)
    if not delta_records:
        # Nothing new to train on — exit 0 (not an error)
        log.info("No new training examples. Exiting cleanly.")
        sys.exit(0)

    # 3. Load model + tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.base_model, args.cpu)
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        sys.exit(2)

    # 4. Build dataset from delta records only
    try:
        dataset = build_hf_dataset(delta_records, tokenizer, args.max_seq_len)
    except Exception as exc:
        log.error("Failed to build dataset: %s", exc)
        sys.exit(2)

    # 5. Train
    try:
        train(model, tokenizer, dataset, args.output, args.epochs, args.cpu)
    except Exception as exc:
        log.error("Training failed: %s", exc)
        sys.exit(2)

    # 6. Record new trained count for next incremental run
    write_trained_count(args.output, new_total)
    # 7. Write training metadata README
    _write_readme(args.output, new_total, len(delta_records))
    # 8. Create/update symlink: models/latest -> claim-verifier-latest/
    _update_latest_symlink(args.output)
    log.info("Pipeline complete. Adapter at: %s", args.output)


def _write_readme(output_path: str, total_examples: int, delta_examples: int) -> None:
    """Write a README.md with training metadata to the adapter directory."""
    output_dir = Path(output_path)
    readme_path = output_dir / "README.md"
    now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    content = (
        f"# claim-verifier LoRA Adapter\n\n"
        f"Auto-generated by finetunePipeline.py\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| Base model | {BASE_MODEL_ID} |\n"
        f"| LoRA r | {LORA_R} |\n"
        f"| LoRA alpha | {LORA_ALPHA} |\n"
        f"| Epochs | {NUM_EPOCHS} |\n"
        f"| Training examples (total) | {total_examples} |\n"
        f"| New examples (this run) | {delta_examples} |\n"
        f"| Generated at | {now} |\n"
    )
    readme_path.write_text(content, encoding="utf-8")
    log.info("README written: %s", readme_path)


def _update_latest_symlink(output_path: str) -> None:
    """Create/update slm-infra-deploy/models/latest -> output_path symlink."""
    output_dir = Path(output_path).resolve()
    models_dir = output_dir.parent
    latest_link = models_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(output_dir)
    log.info("Symlink updated: %s -> %s", latest_link, output_dir)


if __name__ == "__main__":
    main()
