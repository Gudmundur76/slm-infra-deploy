"""
test_cortex_ornith.py

Tests for the Ornith-1.0 routing additions to cortex.py:
  - _ornith_generate: calls /v1/chat/completions, strips <think> blocks, parses JSON
  - _generate: routes to _ollama_generate or _ornith_generate based on LLM_PROVIDER
  - _VerifyHandler: returns 200 with ornith_slm provider, 503 on error
"""
from __future__ import annotations

import io
import json
import os
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import cortex as cx


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_openai_response(content: str) -> MagicMock:
    """Build a fake urllib response that returns an OpenAI-compatible JSON body."""
    body = {
        "choices": [
            {"message": {"content": content, "role": "assistant"}}
        ]
    }
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


VALID_VERDICT = json.dumps({
    "verdict": "Supported",
    "confidence": 0.9,
    "reasoning": "Multiple RCTs confirm the effect.",
    "sources": [{"database": "PubMed", "id": "12345", "url": "https://pubmed.ncbi.nlm.nih.gov/12345"}],
})


# ── _ornith_generate ───────────────────────────────────────────────────────────

class TestOrnithGenerate:
    def test_returns_parsed_verdict_dict(self) -> None:
        """Happy path: valid JSON response → dict returned."""
        with patch("urllib.request.urlopen", return_value=_make_openai_response(VALID_VERDICT)):
            result = cx._ornith_generate("Creatine improves cognition.")
        assert result["verdict"] == "Supported"
        assert result["confidence"] == pytest.approx(0.9)

    def test_strips_think_block_before_parsing(self) -> None:
        """<think>...</think> prefix must be stripped before JSON parse."""
        content_with_think = f"<think>Let me reason step by step...</think>\n{VALID_VERDICT}"
        with patch("urllib.request.urlopen", return_value=_make_openai_response(content_with_think)):
            result = cx._ornith_generate("Vitamin D prevents cancer.")
        assert result["verdict"] == "Supported"

    def test_strips_markdown_code_fence(self) -> None:
        """```json ... ``` wrapping must be stripped before JSON parse."""
        fenced = f"```json\n{VALID_VERDICT}\n```"
        with patch("urllib.request.urlopen", return_value=_make_openai_response(fenced)):
            result = cx._ornith_generate("Omega-3 reduces inflammation.")
        assert result["verdict"] == "Supported"

    def test_strips_think_block_and_code_fence_together(self) -> None:
        """Both <think> block and ``` fence can appear together."""
        content = f"<think>reasoning</think>\n```json\n{VALID_VERDICT}\n```"
        with patch("urllib.request.urlopen", return_value=_make_openai_response(content)):
            result = cx._ornith_generate("Protein timing matters.")
        assert result["verdict"] == "Supported"

    def test_raises_on_invalid_json(self) -> None:
        """If the model returns non-JSON, json.JSONDecodeError is raised."""
        with patch("urllib.request.urlopen", return_value=_make_openai_response("not json")):
            with pytest.raises(json.JSONDecodeError):
                cx._ornith_generate("Bad claim.")

    def test_raises_on_network_error(self) -> None:
        """URLError propagates to the caller."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            with pytest.raises(urllib.error.URLError):
                cx._ornith_generate("Any claim.")

    def test_uses_ornith_url_from_env(self) -> None:
        """ORNITH_SLM_URL env var is used as the base URL."""
        captured: list[Any] = []

        def fake_urlopen(req: Any, timeout: int = 60):
            captured.append(req.full_url)
            return _make_openai_response(VALID_VERDICT)

        with patch.dict(os.environ, {"ORNITH_SLM_URL": "http://custom-host:9999"}):
            # Re-read the module-level constant by patching cx.ORNITH_URL
            with patch.object(cx, "ORNITH_URL", "http://custom-host:9999"):
                with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    cx._ornith_generate("Test claim.")

        assert any("custom-host:9999" in url for url in captured)

    def test_sends_authorization_header(self) -> None:
        """Authorization: Bearer header is included in the request."""
        captured_headers: list[dict] = []

        def fake_urlopen(req: Any, timeout: int = 60):
            captured_headers.append(dict(req.headers))
            return _make_openai_response(VALID_VERDICT)

        with patch.dict(os.environ, {"ORNITH_SLM_API_KEY": "test-key-abc"}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                cx._ornith_generate("Test claim.")

        assert any(
            "Bearer test-key-abc" in str(h.get("Authorization", ""))
            for h in captured_headers
        )

    def test_verdict_fields_present(self) -> None:
        """All required verdict fields are present in the parsed result."""
        with patch("urllib.request.urlopen", return_value=_make_openai_response(VALID_VERDICT)):
            result = cx._ornith_generate("Claim text.")
        for field in ("verdict", "confidence", "reasoning", "sources"):
            assert field in result, f"Missing field: {field}"


# ── _generate router ───────────────────────────────────────────────────────────

class TestGenerateRouter:
    def test_routes_to_ollama_by_default(self) -> None:
        """Without LLM_PROVIDER set, _generate calls _ollama_generate."""
        with patch.object(cx, "LLM_PROVIDER", "ollama"):
            with patch.object(cx, "_ollama_generate", return_value={"verdict": "Supported"}) as mock_ollama:
                with patch.object(cx, "_ornith_generate") as mock_ornith:
                    result = cx._generate("test claim")
        mock_ollama.assert_called_once_with("test claim")
        mock_ornith.assert_not_called()
        assert result["verdict"] == "Supported"

    def test_routes_to_ornith_when_provider_is_ornith_slm(self) -> None:
        """LLM_PROVIDER=ornith_slm routes to _ornith_generate."""
        with patch.object(cx, "LLM_PROVIDER", "ornith_slm"):
            with patch.object(cx, "_ornith_generate", return_value={"verdict": "Contradicted"}) as mock_ornith:
                with patch.object(cx, "_ollama_generate") as mock_ollama:
                    result = cx._generate("test claim")
        mock_ornith.assert_called_once_with("test claim")
        mock_ollama.assert_not_called()
        assert result["verdict"] == "Contradicted"

    def test_unknown_provider_falls_back_to_ollama(self) -> None:
        """Any unrecognised LLM_PROVIDER value falls back to Ollama."""
        with patch.object(cx, "LLM_PROVIDER", "unknown_provider"):
            with patch.object(cx, "_ollama_generate", return_value={"verdict": "Ambiguous"}) as mock_ollama:
                result = cx._generate("test claim")
        mock_ollama.assert_called_once()
        assert result["verdict"] == "Ambiguous"


# ── _VerifyHandler with ornith_slm ─────────────────────────────────────────────

class TestVerifyHandlerOrnith:
    """Integration tests for the HTTP handler using the ornith_slm provider."""

    def _make_handler(self, body: bytes) -> cx._VerifyHandler:
        """Build a _VerifyHandler with a fake request body."""
        handler = cx._VerifyHandler.__new__(cx._VerifyHandler)
        handler.path = "/verify"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_returns_200_with_ornith_verdict(self) -> None:
        """Handler returns 200 when _generate succeeds via ornith_slm."""
        body = json.dumps({"claim": "Protein X causes cancer."}).encode()
        handler = self._make_handler(body)

        with patch.object(cx, "_generate", return_value=json.loads(VALID_VERDICT)):
            handler.do_POST()

        handler.send_response.assert_called_once_with(200)
        written = handler.wfile.getvalue()
        result = json.loads(written)
        assert result["verdict"] == "Supported"

    def test_returns_503_when_ornith_unavailable(self) -> None:
        """Handler returns 503 with fallback=True when _generate raises."""
        body = json.dumps({"claim": "Some claim."}).encode()
        handler = self._make_handler(body)

        with patch.object(cx, "_generate", side_effect=urllib.error.URLError("connection refused")):
            handler.do_POST()

        handler.send_response.assert_called_once_with(503)
        written = handler.wfile.getvalue()
        result = json.loads(written)
        assert result["fallback"] is True
        assert "error" in result
