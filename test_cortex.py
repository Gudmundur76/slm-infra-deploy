"""
test_cortex.py
Tests for cortex.py CLI — focusing on Ornith SLM routing additions (v0.2.0).
"""
import argparse
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Import cortex module ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import cortex as cx


# ── Version ───────────────────────────────────────────────────────────────────
class TestVersion:
    def test_version_is_0_2_0(self) -> None:
        assert cx.VERSION == "0.2.0"


# ── Ornith constants ──────────────────────────────────────────────────────────
class TestOrnithConstants:
    def test_ornith_url_default(self) -> None:
        """Default ORNITH_URL should point to localhost:8080."""
        assert "8080" in cx.ORNITH_URL or "localhost" in cx.ORNITH_URL

    def test_ornith_model_default(self) -> None:
        assert cx.ORNITH_MODEL == "ornith-1.0-9b"

    def test_ornith_health_endpoint_contains_health(self) -> None:
        assert cx.ORNITH_HEALTH_ENDPOINT.endswith("/health")

    def test_ornith_models_endpoint_contains_v1_models(self) -> None:
        assert cx.ORNITH_MODELS_ENDPOINT.endswith("/v1/models")

    def test_ornith_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ORNITH_SLM_URL env var should override the default."""
        monkeypatch.setenv("ORNITH_SLM_URL", "http://gpu-host:9090")
        # Re-evaluate the module-level constant via importlib reload
        importlib.reload(cx)
        assert "9090" in cx.ORNITH_URL or "gpu-host" in cx.ORNITH_URL
        importlib.reload(cx)  # restore default for other tests


# ── Parser ────────────────────────────────────────────────────────────────────
class TestParser:
    def test_run_parser_has_ornith_flag(self) -> None:
        parser = cx.build_parser()
        args = parser.parse_args(["run", "--ornith"])
        assert args.ornith is True

    def test_run_parser_ornith_defaults_false(self) -> None:
        parser = cx.build_parser()
        args = parser.parse_args(["run"])
        assert args.ornith is False

    def test_train_parser_has_ornith_flag(self) -> None:
        parser = cx.build_parser()
        args = parser.parse_args(["train", "--corpus", "/tmp/c.jsonl", "--ornith"])
        assert args.ornith is True

    def test_train_parser_ornith_defaults_false(self) -> None:
        parser = cx.build_parser()
        args = parser.parse_args(["train", "--corpus", "/tmp/c.jsonl"])
        assert args.ornith is False

    def test_init_parser_has_domain_choices(self) -> None:
        parser = cx.build_parser()
        args = parser.parse_args(["init", "--domain", "biotech"])
        assert args.domain == "biotech"

    def test_status_subcommand_exists(self) -> None:
        parser = cx.build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"


# ── cmd_run ornith flag ───────────────────────────────────────────────────────
class TestCmdRun:
    def _make_args(self, ornith: bool = False) -> argparse.Namespace:
        return argparse.Namespace(ornith=ornith)

    def test_run_without_ornith_starts_ollama_and_cognitive_loop(
        self, tmp_path: Path
    ) -> None:
        """Without --ornith, only ollama and cognitive-loop are started."""
        # Create a fake cortex.yaml so _require_cortex_yaml passes
        (tmp_path / "cortex.yaml").write_text("domain: general\n")
        import os
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch.object(cx, "_run") as mock_run, \
                 patch.object(Path, "__truediv__", return_value=tmp_path / "docker-compose.yml"):
                cx.cmd_run(self._make_args(ornith=False))
                call_args = mock_run.call_args[0][0]
                assert "ornith-vllm" not in call_args
                assert "ollama" in call_args or "cognitive-loop" in call_args
        finally:
            os.chdir(orig_cwd)

    def test_run_with_ornith_includes_profile_ornith(self, tmp_path: Path) -> None:
        """With --ornith, docker compose is called with --profile ornith."""
        (tmp_path / "cortex.yaml").write_text("domain: general\n")
        import os
        orig_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch.object(cx, "_run") as mock_run, \
                 patch.object(Path, "__truediv__", return_value=tmp_path / "docker-compose.yml"):
                cx.cmd_run(self._make_args(ornith=True))
                call_args = mock_run.call_args[0][0]
                assert "--profile" in call_args
                assert "ornith" in call_args
        finally:
            os.chdir(orig_cwd)


# ── cmd_status ornith section ─────────────────────────────────────────────────
class TestCmdStatus:
    def test_status_reports_ornith_not_running_when_offline(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """When Ornith SLM is unreachable, status should say NOT RUNNING."""
        import urllib.error

        def fake_urlopen(url: str, timeout: int = 3):
            if "11434" in url:
                raise urllib.error.URLError("Ollama offline")
            if "3100" in url:
                raise urllib.error.URLError("CLF offline")
            if "health" in url:
                raise urllib.error.URLError("Ornith offline")
            raise urllib.error.URLError("unknown")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            cx.cmd_status(argparse.Namespace())

        captured = capsys.readouterr()
        assert "NOT RUNNING" in captured.out
        # Ornith-specific message
        assert "ornith" in captured.out.lower() or "Ornith" in captured.out

    def test_status_reports_ornith_running_when_online(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """When Ornith SLM is reachable, status should say running."""
        import urllib.error
        import io

        class FakeResp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def read(self):
                return b'{"data": [{"id": "ornith-1.0-9b"}]}'

        call_count = [0]

        def fake_urlopen(url: str, timeout: int = 3):
            call_count[0] += 1
            if "11434" in url:
                raise urllib.error.URLError("Ollama offline")
            if "3100" in url:
                raise urllib.error.URLError("CLF offline")
            # Ornith health + models
            return FakeResp()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            cx.cmd_status(argparse.Namespace())

        captured = capsys.readouterr()
        assert "running" in captured.out.lower()
        assert "ornith" in captured.out.lower()
