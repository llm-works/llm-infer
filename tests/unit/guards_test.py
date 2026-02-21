"""Unit tests for generation guards."""

import pytest

from llm_infer.engines.native.guards.protocol import GuardResult
from llm_infer.engines.native.guards.repetition import RepetitionGuard

pytestmark = pytest.mark.unit


class TestGuardResult:
    """Test GuardResult dataclass."""

    def test_continue_action(self) -> None:
        """Test creating a continue result."""
        result = GuardResult("continue")
        assert result.action == "continue"
        assert result.message is None

    def test_stop_action_with_message(self) -> None:
        """Test creating a stop result with message."""
        result = GuardResult("stop", "Test message")
        assert result.action == "stop"
        assert result.message == "Test message"

    def test_warn_action(self) -> None:
        """Test creating a warn result."""
        result = GuardResult("warn", "Warning message")
        assert result.action == "warn"
        assert result.message == "Warning message"


class TestRepetitionGuard:
    """Test RepetitionGuard."""

    def test_default_threshold(self) -> None:
        """Test default threshold is 5."""
        guard = RepetitionGuard()
        assert guard.threshold == 5

    def test_custom_threshold(self) -> None:
        """Test custom threshold."""
        guard = RepetitionGuard(threshold=3)
        assert guard.threshold == 3

    def test_no_repetition_continues(self) -> None:
        """Test that varied tokens continue."""
        guard = RepetitionGuard(threshold=3)
        result = guard.check([1, 2, 3, 4, 5], prompt_tokens=[])
        assert result.action == "continue"

    def test_repetition_detected_stops(self) -> None:
        """Test that repeated tokens trigger stop."""
        guard = RepetitionGuard(threshold=3)
        result = guard.check([1, 2, 5, 5, 5], prompt_tokens=[])
        assert result.action == "stop"
        assert "repetition" in result.message.lower()
        assert "5" in result.message

    def test_not_enough_tokens_continues(self) -> None:
        """Test that short sequences continue."""
        guard = RepetitionGuard(threshold=5)
        result = guard.check([1, 1, 1], prompt_tokens=[])
        assert result.action == "continue"

    def test_exactly_threshold_repetition_stops(self) -> None:
        """Test exactly threshold repetitions triggers stop."""
        guard = RepetitionGuard(threshold=4)
        result = guard.check([1, 2, 3, 3, 3, 3], prompt_tokens=[])
        assert result.action == "stop"

    def test_partial_repetition_continues(self) -> None:
        """Test that partial repetition does not trigger stop."""
        guard = RepetitionGuard(threshold=5)
        result = guard.check([1, 1, 1, 1, 2], prompt_tokens=[])
        assert result.action == "continue"

    def test_empty_tokens_continues(self) -> None:
        """Test that empty token list continues."""
        guard = RepetitionGuard(threshold=3)
        result = guard.check([], prompt_tokens=[])
        assert result.action == "continue"

    def test_single_token_continues(self) -> None:
        """Test that single token continues."""
        guard = RepetitionGuard(threshold=3)
        result = guard.check([42], prompt_tokens=[])
        assert result.action == "continue"
