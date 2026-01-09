"""Unit tests for generation helper functions."""

import pytest
import torch

from llm_infer.primitives.guards import GuardResult, RepetitionGuard

pytestmark = pytest.mark.unit


class TestGuardResultDataclass:
    """Test GuardResult dataclass."""

    def test_continue_action(self) -> None:
        """Test continue action result."""
        result = GuardResult(action="continue")
        assert result.action == "continue"
        assert result.message is None

    def test_stop_action_with_message(self) -> None:
        """Test stop action with message."""
        result = GuardResult(action="stop", message="Too many tokens")
        assert result.action == "stop"
        assert result.message == "Too many tokens"

    def test_warn_action(self) -> None:
        """Test warn action."""
        result = GuardResult(action="warn", message="Low confidence")
        assert result.action == "warn"
        assert result.message == "Low confidence"


class TestRepetitionGuardWithLogits:
    """Test RepetitionGuard with logits parameter."""

    def test_check_ignores_logits(self) -> None:
        """Test that check works regardless of logits."""
        guard = RepetitionGuard(threshold=3)

        # With no repetition, logits are ignored
        logits = torch.randn(1, 100)
        result = guard.check([1, 2, 3], [10, 20], logits)
        assert result.action == "continue"

    def test_check_with_none_logits(self) -> None:
        """Test that check works with None logits."""
        guard = RepetitionGuard(threshold=3)
        result = guard.check([1, 2, 3], [10, 20], None)
        assert result.action == "continue"

    def test_check_detects_repetition_with_logits(self) -> None:
        """Test repetition detection even with logits provided."""
        guard = RepetitionGuard(threshold=3)
        logits = torch.randn(1, 100)
        # 3 repetitions of token 5
        result = guard.check([5, 5, 5], [10], logits)
        assert result.action == "stop"
