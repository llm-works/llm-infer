"""Unit tests for FallbackClient and fallback helpers."""

from unittest.mock import MagicMock

import pytest

from llm_infer.client.fallback_helper import build_model_chain, detect_cycles

pytestmark = pytest.mark.unit


class TestBuildModelChain:
    """Tests for build_model_chain helper."""

    def test_no_fallback_returns_single_model(self) -> None:
        """Model without fallback returns just that model."""
        fallbacks: dict[str, str] = {}
        chain = build_model_chain("gpt-4o", fallbacks)
        assert chain == ["gpt-4o"]

    def test_single_fallback(self) -> None:
        """Single fallback pair creates two-model chain."""
        fallbacks = {"gpt-4o": "claude-sonnet"}
        chain = build_model_chain("gpt-4o", fallbacks)
        assert chain == ["gpt-4o", "claude-sonnet"]

    def test_chained_fallbacks(self) -> None:
        """Fallback pairs chain together."""
        fallbacks = {
            "gpt-4o": "claude-sonnet",
            "claude-sonnet": "gemini-pro",
        }
        chain = build_model_chain("gpt-4o", fallbacks)
        assert chain == ["gpt-4o", "claude-sonnet", "gemini-pro"]

    def test_none_model_returns_none_list(self) -> None:
        """None model returns [None]."""
        fallbacks = {"gpt-4o": "claude-sonnet"}
        chain = build_model_chain(None, fallbacks)
        assert chain == [None]

    def test_model_not_in_fallbacks(self) -> None:
        """Model not in fallbacks returns just that model."""
        fallbacks = {"other-model": "fallback"}
        chain = build_model_chain("gpt-4o", fallbacks)
        assert chain == ["gpt-4o"]

    def test_cycle_stops_chain(self) -> None:
        """Cycle in fallbacks stops chain building."""
        fallbacks = {
            "a": "b",
            "b": "a",
        }
        chain = build_model_chain("a", fallbacks)
        assert chain == ["a", "b"]


class TestDetectCycles:
    """Tests for detect_cycles helper."""

    def test_no_cycles_returns_empty(self) -> None:
        """Config without cycles returns empty set."""
        fallbacks = {
            "gpt-4o": "claude-sonnet",
            "claude-sonnet": "gemini-pro",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == set()
        lg.warning.assert_not_called()

    def test_simple_cycle_detected(self) -> None:
        """Simple A->B->A cycle is detected."""
        fallbacks = {
            "a": "b",
            "b": "a",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert "a" in cycles or "b" in cycles
        lg.warning.assert_called_once()
        call_args = lg.warning.call_args
        assert "cycle" in call_args[1]["extra"]

    def test_longer_cycle_detected(self) -> None:
        """Longer A->B->C->A cycle is detected."""
        fallbacks = {
            "a": "b",
            "b": "c",
            "c": "a",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert len(cycles) > 0
        lg.warning.assert_called()

    def test_self_loop_detected(self) -> None:
        """Self-loop A->A is detected."""
        fallbacks = {"a": "a"}
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert "a" in cycles
        lg.warning.assert_called_once()


class TestFallbackClientImport:
    """Test FallbackClient can be imported."""

    def test_import_from_client_package(self) -> None:
        """FallbackClient is exported from client package."""
        from llm_infer.client import FallbackClient

        assert FallbackClient is not None

    def test_import_directly(self) -> None:
        """FallbackClient can be imported directly."""
        from llm_infer.client.fallback import FallbackClient

        assert FallbackClient is not None
