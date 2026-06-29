"""
test_oumiPipeline.py

Tests for oumiPipeline.py — the Oumi-based SLM fine-tuning pipeline.
Covers: load_corpus, convert_to_oumi_dataset, build_oumi_config,
        run_oumi_train (mocked), and the main() fallback behaviour.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent


def make_corpus(path: Path, n: int = 12, valid: bool = True) -> None:
    """Write n valid Alpaca-format JSONL records to path."""
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            if valid:
                rec = {
                    "instruction": f"Verify claim {i}",
                    "input": f"Context {i}",
                    "output": f"Supported with confidence 0.{i % 10}",
                }
            else:
                f.write("not json\n")
                continue
            f.write(json.dumps(rec) + "\n")


# ── load_corpus ───────────────────────────────────────────────────────────────

class TestLoadCorpus:
    def test_loads_valid_corpus(self, tmp_path):
        from oumiPipeline import load_corpus
        corpus = tmp_path / "corpus.jsonl"
        make_corpus(corpus, n=15)
        records = load_corpus(str(corpus))
        assert len(records) == 15
        assert all("instruction" in r for r in records)

    def test_raises_on_missing_file(self, tmp_path):
        from oumiPipeline import load_corpus
        with pytest.raises(SystemExit) as exc:
            load_corpus(str(tmp_path / "nonexistent.jsonl"))
        assert exc.value.code == 1

    def test_raises_on_corpus_too_small(self, tmp_path):
        from oumiPipeline import load_corpus
        corpus = tmp_path / "small.jsonl"
        make_corpus(corpus, n=5)
        with pytest.raises(SystemExit) as exc:
            load_corpus(str(corpus))
        assert exc.value.code == 1

    def test_skips_malformed_json_lines(self, tmp_path):
        from oumiPipeline import load_corpus
        corpus = tmp_path / "mixed.jsonl"
        with corpus.open("w") as f:
            # Write 10 valid + 5 malformed
            for i in range(10):
                f.write(json.dumps({"instruction": f"i{i}", "input": "", "output": f"o{i}"}) + "\n")
            for _ in range(5):
                f.write("not json\n")
        records = load_corpus(str(corpus))
        assert len(records) == 10

    def test_skips_records_missing_required_keys(self, tmp_path):
        from oumiPipeline import load_corpus
        corpus = tmp_path / "partial.jsonl"
        with corpus.open("w") as f:
            # 10 valid records
            for i in range(10):
                f.write(json.dumps({"instruction": f"i{i}", "input": "", "output": f"o{i}"}) + "\n")
            # 3 records missing 'output'
            for i in range(3):
                f.write(json.dumps({"instruction": f"bad{i}", "input": ""}) + "\n")
        records = load_corpus(str(corpus))
        assert len(records) == 10

    def test_skips_empty_lines(self, tmp_path):
        from oumiPipeline import load_corpus
        corpus = tmp_path / "empty_lines.jsonl"
        with corpus.open("w") as f:
            for i in range(10):
                f.write(json.dumps({"instruction": f"i{i}", "input": "", "output": f"o{i}"}) + "\n")
                f.write("\n")  # blank line between each record
        records = load_corpus(str(corpus))
        assert len(records) == 10


# ── convert_to_oumi_dataset ───────────────────────────────────────────────────

class TestConvertToOumiDataset:
    def test_creates_train_jsonl(self, tmp_path):
        from oumiPipeline import convert_to_oumi_dataset
        records = [
            {"instruction": "Verify", "input": "Context", "output": "Supported"},
            {"instruction": "Check", "input": "Data", "output": "Contradicted"},
        ]
        dataset_dir = tmp_path / "dataset"
        convert_to_oumi_dataset(records, dataset_dir)
        assert (dataset_dir / "train.jsonl").exists()

    def test_train_jsonl_has_messages_format(self, tmp_path):
        from oumiPipeline import convert_to_oumi_dataset
        records = [{"instruction": "Verify X", "input": "Context Y", "output": "Supported"}]
        dataset_dir = tmp_path / "dataset"
        convert_to_oumi_dataset(records, dataset_dir)
        lines = (dataset_dir / "train.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert "messages" in row
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"
        assert row["messages"][1]["content"] == "Supported"

    def test_prompt_contains_instruction_and_input(self, tmp_path):
        from oumiPipeline import convert_to_oumi_dataset
        records = [{"instruction": "Check claim", "input": "Some context", "output": "OK"}]
        dataset_dir = tmp_path / "dataset"
        convert_to_oumi_dataset(records, dataset_dir)
        row = json.loads((dataset_dir / "train.jsonl").read_text())
        user_content = row["messages"][0]["content"]
        assert "Check claim" in user_content
        assert "Some context" in user_content

    def test_creates_dataset_info_json(self, tmp_path):
        from oumiPipeline import convert_to_oumi_dataset
        records = [{"instruction": f"i{i}", "input": "", "output": f"o{i}"} for i in range(3)]
        dataset_dir = tmp_path / "dataset"
        convert_to_oumi_dataset(records, dataset_dir)
        info = json.loads((dataset_dir / "dataset_info.json").read_text())
        assert info["splits"]["train"]["num_examples"] == 3

    def test_handles_empty_input_field(self, tmp_path):
        from oumiPipeline import convert_to_oumi_dataset
        records = [{"instruction": "Verify", "input": "", "output": "Supported"}]
        dataset_dir = tmp_path / "dataset"
        convert_to_oumi_dataset(records, dataset_dir)
        row = json.loads((dataset_dir / "train.jsonl").read_text())
        # Should not crash and should produce a valid messages record
        assert row["messages"][1]["content"] == "Supported"


# ── build_oumi_config ─────────────────────────────────────────────────────────

class TestBuildOumiConfig:
    def test_replaces_corpus_dataset_path_placeholder(self, tmp_path):
        from oumiPipeline import build_oumi_config
        template = tmp_path / "template.yaml"
        template.write_text("dataset_name: __CORPUS_DATASET_PATH__\noutput_dir: __OUTPUT_DIR__\n")
        result_path = build_oumi_config(
            str(template),
            "/data/dataset",
            "/data/output",
            tmp_path,
        )
        content = Path(result_path).read_text()
        assert "__CORPUS_DATASET_PATH__" not in content
        assert "/data/dataset" in content

    def test_replaces_output_dir_placeholder(self, tmp_path):
        from oumiPipeline import build_oumi_config
        template = tmp_path / "template.yaml"
        template.write_text("output_dir: __OUTPUT_DIR__\n")
        result_path = build_oumi_config(
            str(template),
            "/data/dataset",
            "/data/output",
            tmp_path,
        )
        content = Path(result_path).read_text()
        assert "__OUTPUT_DIR__" not in content
        assert "/data/output" in content

    def test_writes_resolved_config_to_tmp_dir(self, tmp_path):
        from oumiPipeline import build_oumi_config
        template = tmp_path / "t.yaml"
        template.write_text("x: __CORPUS_DATASET_PATH__\n")
        result_path = build_oumi_config(str(template), "/d", "/o", tmp_path)
        assert Path(result_path).parent == tmp_path
        assert Path(result_path).name == "oumi_train_resolved.yaml"

    def test_real_config_template_has_placeholders(self):
        """The actual oumi_cpu_train.yaml must contain both placeholders."""
        config_path = SCRIPT_DIR / "configs" / "oumi_cpu_train.yaml"
        if not config_path.exists():
            pytest.skip("oumi_cpu_train.yaml not found")
        content = config_path.read_text()
        assert "__CORPUS_DATASET_PATH__" in content, "Missing __CORPUS_DATASET_PATH__ placeholder"
        assert "__OUTPUT_DIR__" in content, "Missing __OUTPUT_DIR__ placeholder"


# ── run_oumi_train ────────────────────────────────────────────────────────────

class TestRunOumiTrain:
    def test_calls_oumi_train_with_config(self, tmp_path):
        from oumiPipeline import run_oumi_train
        config = tmp_path / "config.yaml"
        config.write_text("model: test\n")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_oumi_train(config)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "oumi"
        assert cmd[1] == "train"
        assert cmd[2] == "-c"
        assert cmd[3] == str(config)

    def test_raises_runtime_error_on_nonzero_exit(self, tmp_path):
        from oumiPipeline import run_oumi_train
        config = tmp_path / "config.yaml"
        config.write_text("model: test\n")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=2)
            with pytest.raises(RuntimeError, match="oumi train exited with code 2"):
                run_oumi_train(config)


# ── run_fallback_pipeline ─────────────────────────────────────────────────────

class TestRunFallbackPipeline:
    def test_calls_finetunePipeline_when_oumi_missing(self, tmp_path):
        from oumiPipeline import run_fallback_pipeline
        fallback = SCRIPT_DIR / "finetunePipeline.py"
        if not fallback.exists():
            pytest.skip("finetunePipeline.py not found")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            run_fallback_pipeline(str(tmp_path / "corpus.jsonl"), str(tmp_path / "out"), cpu=True)
        cmd = mock_run.call_args[0][0]
        assert "finetunePipeline.py" in cmd[1]
        assert "--corpus" in cmd
        assert "--cpu" in cmd

    def test_exits_3_when_neither_available(self, tmp_path):
        from oumiPipeline import run_fallback_pipeline
        with patch("oumiPipeline.SCRIPT_DIR", tmp_path):
            with pytest.raises(SystemExit) as exc:
                run_fallback_pipeline(str(tmp_path / "c.jsonl"), str(tmp_path / "o"), cpu=False)
        assert exc.value.code == 3

    def test_exits_with_fallback_returncode_on_failure(self, tmp_path):
        from oumiPipeline import run_fallback_pipeline
        fallback = SCRIPT_DIR / "finetunePipeline.py"
        if not fallback.exists():
            pytest.skip("finetunePipeline.py not found")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=5)
            with pytest.raises(SystemExit) as exc:
                run_fallback_pipeline(str(tmp_path / "c.jsonl"), str(tmp_path / "o"), cpu=False)
        assert exc.value.code == 5


# ── main() integration ────────────────────────────────────────────────────────

class TestMain:
    def test_main_uses_fallback_when_oumi_not_installed(self, tmp_path):
        """When oumi CLI is absent, main() should call run_fallback_pipeline."""
        corpus = tmp_path / "corpus.jsonl"
        make_corpus(corpus, n=12)
        output = tmp_path / "output"
        with patch("shutil.which", return_value=None), \
             patch("oumiPipeline.run_fallback_pipeline") as mock_fallback:
            sys.argv = ["oumiPipeline.py", "--corpus", str(corpus), "--output", str(output)]
            from oumiPipeline import main
            main()
        mock_fallback.assert_called_once()

    def test_main_runs_oumi_when_installed(self, tmp_path):
        """When oumi CLI is present, main() should call run_oumi_train."""
        corpus = tmp_path / "corpus.jsonl"
        make_corpus(corpus, n=12)
        output = tmp_path / "output"
        with patch("shutil.which", return_value="/usr/bin/oumi"), \
             patch("oumiPipeline.run_oumi_train") as mock_train:
            sys.argv = ["oumiPipeline.py", "--corpus", str(corpus), "--output", str(output)]
            from oumiPipeline import main
            main()
        mock_train.assert_called_once()

    def test_main_exits_1_on_missing_corpus(self, tmp_path):
        output = tmp_path / "output"
        with patch("shutil.which", return_value="/usr/bin/oumi"):
            sys.argv = [
                "oumiPipeline.py",
                "--corpus", str(tmp_path / "nonexistent.jsonl"),
                "--output", str(output),
            ]
            with pytest.raises(SystemExit) as exc:
                from oumiPipeline import main
                main()
        assert exc.value.code == 1
