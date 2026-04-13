"""Unit tests for cli/tools/metrics.py MetricsTool."""

from __future__ import annotations

import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_infer.cli.tools.metrics import MetricsTool

pytestmark = pytest.mark.unit


def _make_tool(**args_overrides: Any) -> MetricsTool:
    """Create a MetricsTool with mocked args + lg."""
    import argparse

    tool = MetricsTool.__new__(MetricsTool)
    tool._parsed_args = argparse.Namespace(
        host=args_overrides.get("host", "localhost"),
        port=args_overrides.get("port", 8000),
        reset_peak=args_overrides.get("reset_peak", False),
        json=args_overrides.get("json", False),
    )
    tool._logger = MagicMock()
    return tool


def _metrics_response() -> dict:
    return {
        "gpu": {
            "allocated_mb": 1024.5,
            "reserved_mb": 2048.0,
            "peak_mb": 3072.0,
        },
        "kv_cache": {
            "mb": 512.0,
            "blocks_used": 10,
            "blocks_total": 100,
            "capacity_tokens": 1600,
        },
        "sequences": {"active": 2, "total_tokens": 200},
        "pending_requests": 3,
    }


class _FakeResponse:
    """Fake urlopen response context manager."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class TestMetricsTool:
    def test_add_args(self) -> None:
        import argparse

        tool = MetricsTool.__new__(MetricsTool)
        parser = argparse.ArgumentParser()
        tool.add_args(parser)
        # Verify arguments were added
        args = parser.parse_args([])
        assert args.host == "localhost"
        assert args.port == 8000
        assert args.reset_peak is False
        assert args.json is False

    def test_run_success_formatted(self) -> None:
        tool = _make_tool()
        with patch(
            "urllib.request.urlopen", return_value=_FakeResponse(_metrics_response())
        ):
            result = tool.run()
        assert result == 0
        assert tool.lg.info.called

    def test_run_success_json(self, capsys: pytest.CaptureFixture) -> None:
        tool = _make_tool(json=True)
        with patch(
            "urllib.request.urlopen", return_value=_FakeResponse(_metrics_response())
        ):
            result = tool.run()
        assert result == 0
        captured = capsys.readouterr()
        assert "gpu" in captured.out

    def test_run_with_reset_peak(self) -> None:
        tool = _make_tool(reset_peak=True)
        captured_url = []

        def fake_urlopen(req, timeout):
            captured_url.append(req.full_url)
            return _FakeResponse(_metrics_response())

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = tool.run()
        assert result == 0
        assert "reset_peak=true" in captured_url[0]

    def test_run_connection_refused(self) -> None:
        tool = _make_tool()
        err = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            result = tool.run()
        assert result == 1
        # Check error message is "server not running"
        error_calls = [c.args[0] for c in tool.lg.error.call_args_list if c.args]
        assert "server not running" in error_calls

    def test_run_other_url_error(self) -> None:
        tool = _make_tool()
        err = urllib.error.URLError("DNS lookup failed")
        with patch("urllib.request.urlopen", side_effect=err):
            result = tool.run()
        assert result == 1

    def test_run_generic_exception(self) -> None:
        tool = _make_tool()
        with patch("urllib.request.urlopen", side_effect=ValueError("oops")):
            result = tool.run()
        assert result == 1


def test_metrics_tool_init() -> None:
    """Test the __init__ wires up config correctly."""
    tool = MetricsTool(parent=None)
    assert tool is not None
