"""Repetition detection guard."""

from torch import Tensor

from .protocol import GuardResult


class RepetitionGuard:
    """Detects when the model outputs the same token repeatedly.

    This is a common failure mode when something is wrong with the model
    configuration, tokenization, or when the model gets stuck in a loop.

    Args:
        threshold: Number of identical consecutive tokens to trigger stop.
            Default is 5, meaning if the same token appears 5 times in a row,
            generation stops.
    """

    def __init__(self, threshold: int = 5):
        self.threshold = threshold

    def check(
        self,
        generated_tokens: list[int],
        prompt_tokens: list[int],
        logits: Tensor | None = None,
    ) -> GuardResult:
        """Check for token repetition.

        Args:
            generated_tokens: Tokens generated so far.
            prompt_tokens: Original prompt tokens (unused).
            logits: Logits from last forward pass (unused).

        Returns:
            GuardResult with "stop" action if repetition detected.
        """
        if len(generated_tokens) < self.threshold:
            return GuardResult("continue")

        # Check if last N tokens are all identical
        last_tokens = generated_tokens[-self.threshold :]
        if len(set(last_tokens)) == 1:
            token_id = last_tokens[0]
            return GuardResult(
                "stop",
                f"Token repetition detected: token {token_id} repeated {self.threshold} times",
            )

        return GuardResult("continue")
