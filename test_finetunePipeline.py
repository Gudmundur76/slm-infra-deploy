"""
test_finetunePipeline.py

Pytest test suite for finetunePipeline.py — the LoRA fine-tuning pipeline.

Ralph Wiggum loop: RED → GREEN → VALIDATE → COMPLETE
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))


# ── Mock heavy ML dependencies before import ──────────────────────────────────
_ML_MOCKS = {
    "torch": MagicMock(),
    "transformers": MagicMock(),
    "peft": MagicMock(),
    "datasets": MagicMock(),
}

with patch.dict("sys.modules", _ML_MOCKS):
    import finetunePipeline as fp


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_corpus(tmp_path: Path) -> Path:
    """Create a minimal JSONL corpus with 10 training pairs (minimum required by pipeline)."""
    corpus = tmp_path / "corpus.jsonl"
    base_pairs = [
        {
            "instruction": "Classify the scientific claim.",
            "input": "Protein XYZ binds to receptor ABC.",
            "output": "Supported (confidence: 0.95)",
            "type": "classify",
        },
        {
            "instruction": "Extract all scientific entities.",
            "input": "Protein XYZ binds to receptor ABC.",
            "output": '[{"type": "protein", "name": "XYZ"}]',
            "type": "extract",
        },
        {
            "instruction": "Explain the provenance chain.",
            "input": "Protein XYZ binds to receptor ABC.",
            "output": "Paper 123 -> Supported",
            "type": "provenance",
        },
    ]
    # Pad to 10 pairs to satisfy the minimum corpus size requirement
    pairs = base_pairs + [
        {
            "instruction": f"Classify claim {i}.",
            "input": f"Molecule {i} inhibits enzyme {i}.",
            "output": f"Supported (confidence: 0.{80 + i})",
            "type": "classify",
        }
        for i in range(7)
    ]
    with corpus.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    return corpus


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Provide a temporary output directory for adapter weights."""
    output = tmp_path / "adapter"
    output.mkdir()
    return output


# ── Tests: corpus loading ─────────────────────────────────────────────────────
class TestCorpusLoading:
    def test_loads_all_lines(self, tmp_corpus: Path) -> None:
        pairs = fp.load_corpus(str(tmp_corpus))
        assert len(pairs) == 10

    def test_each_pair_has_required_fields(self, tmp_corpus: Path) -> None:
        pairs = fp.load_corpus(str(tmp_corpus))
        for pair in pairs:
            assert "instruction" in pair
            assert "input" in pair
            assert "output" in pair

    def test_raises_on_missing_corpus(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, SystemExit)):
            fp.load_corpus(str(tmp_path / "nonexistent.jsonl"))

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus_empty_lines.jsonl"
        # Write 10 valid lines plus empty lines to satisfy minimum corpus size
        lines = ""
        for i in range(10):
            lines += json.dumps({"instruction": f"i{i}", "input": f"x{i}", "output": f"y{i}", "type": "classify"}) + "\n"
        lines += "\n\n"  # extra empty lines that should be skipped
        corpus.write_text(lines)
        pairs = fp.load_corpus(str(corpus))
        assert len(pairs) == 10

    def test_skips_malformed_json_lines(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus_malformed.jsonl"
        # Write 10 valid lines plus 1 malformed line
        lines = ""
        for i in range(10):
            lines += json.dumps({"instruction": f"i{i}", "input": f"x{i}", "output": f"y{i}", "type": "classify"}) + "\n"
        lines += "{INVALID}\n"  # malformed line that should be skipped
        corpus.write_text(lines)
        pairs = fp.load_corpus(str(corpus))
        assert len(pairs) == 10


# ── Tests: prompt formatting ──────────────────────────────────────────────────
class TestPromptFormatting:
    def test_formats_alpaca_style_prompt(self) -> None:
        pair = {
            "instruction": "Classify the claim.",
            "input": "Protein XYZ binds to receptor ABC.",
            "output": "Supported",
        }
        prompt = fp.format_prompt(pair)
        assert "### Instruction:" in prompt
        assert "Classify the claim." in prompt
        assert "### Input:" in prompt
        assert "Protein XYZ binds to receptor ABC." in prompt
        assert "### Response:" in prompt

    def test_prompt_contains_response_section(self) -> None:
        pair = {"instruction": "i", "input": "x", "output": "y"}
        prompt = fp.format_prompt(pair)
        assert "### Response:" in prompt

    def test_handles_empty_input_field(self) -> None:
        pair = {"instruction": "Classify.", "input": "", "output": "Supported"}
        # Should not raise
        prompt = fp.format_prompt(pair)
        assert "### Instruction:" in prompt


# ── Tests: incremental delta detection ───────────────────────────────────────
class TestIncrementalDelta:
    def test_returns_all_records_when_no_prior_run(
        self, tmp_corpus: Path, tmp_output: Path
    ) -> None:
        """First run: checkpoint at 0 → all 10 pairs are new."""
        records = fp.load_corpus(str(tmp_corpus))
        delta, total = fp.get_delta_records(records, str(tmp_output))
        assert len(delta) == 10
        assert total == 10

    def test_returns_empty_when_all_trained(
        self, tmp_corpus: Path, tmp_output: Path
    ) -> None:
        """Second run: checkpoint at 10 → no new pairs."""
        records = fp.load_corpus(str(tmp_corpus))
        # Simulate a prior training run that processed all 10 records
        fp.write_trained_count(str(tmp_output), 10)
        delta, total = fp.get_delta_records(records, str(tmp_output))
        assert len(delta) == 0
        assert total == 10

    def test_returns_only_new_records(self, tmp_path: Path) -> None:
        """Third run: checkpoint at 2, 12 total → 10 new pairs."""
        corpus = tmp_path / "corpus.jsonl"
        output = tmp_path / "adapter"
        output.mkdir()

        # Write 12 pairs (above the 10-minimum threshold)
        with corpus.open("w") as f:
            for i in range(12):
                f.write(json.dumps({
                    "instruction": f"inst {i}",
                    "input": f"input {i}",
                    "output": f"output {i}",
                    "type": "classify",
                }) + "\n")

        records = fp.load_corpus(str(corpus))
        # Simulate 2 already trained
        fp.write_trained_count(str(output), 2)
        delta, total = fp.get_delta_records(records, str(output))
        assert len(delta) == 10
        assert total == 12

    def test_trained_count_persists_to_disk(self, tmp_output: Path) -> None:
        """write_trained_count / read_trained_count round-trip."""
        fp.write_trained_count(str(tmp_output), 42)
        count = fp.read_trained_count(str(tmp_output))
        assert count == 42

    def test_read_trained_count_returns_zero_when_missing(
        self, tmp_output: Path
    ) -> None:
        """No checkpoint file → returns 0 (start from scratch)."""
        count = fp.read_trained_count(str(tmp_output))
        assert count == 0


# ── Tests: checkpoint file path ──────────────────────────────────────────────
class TestCheckpointPath:
    def test_checkpoint_path_is_inside_output_dir(self, tmp_output: Path) -> None:
        checkpoint_path = fp.get_trained_count_path(str(tmp_output))
        assert str(tmp_output) in str(checkpoint_path)

    def test_checkpoint_path_is_a_file_path(self, tmp_output: Path) -> None:
        checkpoint_path = fp.get_trained_count_path(str(tmp_output))
        # Should be a Path object pointing to a file (not a directory)
        assert isinstance(checkpoint_path, Path)
        assert checkpoint_path.suffix in (".txt", ".json", "")
