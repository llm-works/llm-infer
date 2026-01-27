"""Generation guard protocol and types."""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from torch import Tensor


@dataclass
class GuardResult:
    """Result from a generation guard check.

    Attributes:
        action: What action to take:
            - "continue": Normal operation, keep generating
            - "warn": Add warning but continue generating
            - "stop": Stop generation immediately
        message: Optional message explaining the result
    """

    action: Literal["continue", "warn", "stop"]
    message: str | None = None


@runtime_checkable
class GenerationGuard(Protocol):
    """Protocol for pluggable generation quality checks.

    Guards are called after each token is generated. They can inspect
    the generated tokens and optionally the logits to detect issues
    like repetition, low confidence, or other quality problems.

    Example implementation:
        class RepetitionGuard:
            def __init__(self, threshold: int = 5):
                self.threshold = threshold

            def check(self, generated_tokens, prompt_tokens, logits=None):
                if len(generated_tokens) >= self.threshold:
                    if len(set(generated_tokens[-self.threshold:])) == 1:
                        return GuardResult("stop", "Token repetition detected")
                return GuardResult("continue")
    """

    def check(
        self,
        generated_tokens: list[int],
        prompt_tokens: list[int],
        logits: Tensor | None = None,
    ) -> GuardResult:
        """Check generation state and return action to take.

        Args:
            generated_tokens: Tokens generated so far (not including prompt).
            prompt_tokens: Original prompt tokens.
            logits: Optional logits from the last forward pass.

        Returns:
            GuardResult indicating what action to take.
        """
        ...
