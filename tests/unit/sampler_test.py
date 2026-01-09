"""Unit tests for token sampling."""

import pytest
import torch

from llm_infer.primitives.sampler.sampler import (
    _apply_repetition_penalty,
    _apply_top_k,
    _apply_top_p,
    sample,
)

pytestmark = pytest.mark.unit


class TestSampleGreedy:
    """Test greedy sampling (temperature=0)."""

    def test_greedy_returns_argmax(self) -> None:
        """Test that temperature=0 returns argmax."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        result = sample(logits, temperature=0)
        assert result.item() == 1  # index of 5.0

    def test_greedy_batch(self) -> None:
        """Test greedy sampling with batch."""
        logits = torch.tensor(
            [
                [1.0, 5.0, 2.0],
                [3.0, 1.0, 2.0],
            ]
        )
        result = sample(logits, temperature=0)
        assert result.tolist() == [1, 0]


class TestSampleTemperature:
    """Test temperature sampling."""

    def test_high_temperature_more_random(self) -> None:
        """Test that high temperature produces more varied outputs."""
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 1.1, 1.2, 1.3]])

        # Sample many times with high temperature
        samples = [sample(logits, temperature=2.0).item() for _ in range(100)]

        # Should see variety (not all the same)
        unique = set(samples)
        assert len(unique) > 1

    def test_low_temperature_less_random(self) -> None:
        """Test that low temperature favors highest logit."""
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 1.0, 1.0, 10.0]])

        # Sample many times with low temperature
        samples = [sample(logits, temperature=0.1).item() for _ in range(20)]

        # Should mostly be index 3 (highest logit)
        assert samples.count(3) > 15


class TestApplyTopK:
    """Test top-k filtering."""

    def test_top_k_masks_low_logits(self) -> None:
        """Test that logits below top-k are masked."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        result = _apply_top_k(logits, top_k=3)

        # Top 3 are indices 1 (5.0), 4 (4.0), 2 (3.0)
        assert result[0, 1] == 5.0  # kept
        assert result[0, 4] == 4.0  # kept
        assert result[0, 2] == 3.0  # kept
        assert result[0, 0] == float("-inf")  # masked
        assert result[0, 3] == float("-inf")  # masked

    def test_top_k_larger_than_vocab(self) -> None:
        """Test that top_k > vocab_size doesn't crash."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = _apply_top_k(logits, top_k=100)
        assert torch.equal(result, logits)


class TestApplyTopP:
    """Test nucleus (top-p) sampling."""

    def test_top_p_masks_low_prob_tokens(self) -> None:
        """Test that top-p masks low probability tokens."""
        # Logits where first token has very high probability
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        result = _apply_top_p(logits, top_p=0.5)

        # First token should be kept
        assert result[0, 0] != float("-inf")
        # At least one low-prob token should be masked
        assert torch.isinf(result[0, 1:]).any()

    def test_top_p_one_keeps_all(self) -> None:
        """Test that top_p=1.0 keeps all tokens."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = _apply_top_p(logits, top_p=1.0)
        # With top_p=1.0, no tokens should be masked
        assert not torch.isinf(result).any()

    def test_top_p_always_keeps_at_least_one(self) -> None:
        """Test that at least one token is always kept."""
        logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        result = _apply_top_p(logits, top_p=0.01)  # Very low top_p

        # At least one token should not be -inf
        assert not torch.isinf(result).all()


class TestApplyRepetitionPenalty:
    """Test repetition penalty."""

    def test_penalty_reduces_positive_logits(self) -> None:
        """Test that penalty reduces positive logits for past tokens."""
        logits = torch.tensor([[4.0, 2.0, 3.0]])
        result = _apply_repetition_penalty(logits, past_tokens=[0], penalty=2.0)

        assert result[0, 0] == 2.0  # 4.0 / 2.0
        assert result[0, 1] == 2.0  # unchanged
        assert result[0, 2] == 3.0  # unchanged

    def test_penalty_increases_negative_logits_magnitude(self) -> None:
        """Test that penalty increases magnitude of negative logits."""
        logits = torch.tensor([[-2.0, 1.0, 3.0]])
        result = _apply_repetition_penalty(logits, past_tokens=[0], penalty=2.0)

        assert result[0, 0] == -4.0  # -2.0 * 2.0
        assert result[0, 1] == 1.0  # unchanged

    def test_penalty_multiple_tokens(self) -> None:
        """Test penalty applied to multiple past tokens."""
        logits = torch.tensor([[4.0, 4.0, 4.0]])
        result = _apply_repetition_penalty(logits, past_tokens=[0, 2], penalty=2.0)

        assert result[0, 0] == 2.0  # penalized
        assert result[0, 1] == 4.0  # unchanged
        assert result[0, 2] == 2.0  # penalized

    def test_no_penalty_when_disabled(self) -> None:
        """Test that penalty=1.0 has no effect."""
        logits = torch.tensor([[4.0, 2.0, 3.0]])
        result = _apply_repetition_penalty(logits, past_tokens=[0, 1], penalty=1.0)

        assert torch.equal(result, logits)


class TestSampleWithOptions:
    """Test sample function with various options combined."""

    def test_sample_with_top_k(self) -> None:
        """Test sampling with top_k enabled."""
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 10.0, 2.0, 9.0]])

        # With top_k=2, only indices 1 and 3 should be possible
        samples = [sample(logits, temperature=1.0, top_k=2).item() for _ in range(50)]
        unique = set(samples)
        assert unique.issubset({1, 3})

    def test_sample_with_repetition_penalty(self) -> None:
        """Test sampling with repetition penalty."""
        torch.manual_seed(42)
        logits = torch.tensor([[5.0, 5.0, 5.0]])

        # Penalize token 0 heavily
        samples_penalized = [
            sample(
                logits, temperature=1.0, repetition_penalty=10.0, past_tokens=[0]
            ).item()
            for _ in range(50)
        ]

        # Token 0 should appear less frequently
        count_0 = samples_penalized.count(0)
        assert count_0 < 25  # Should be significantly less than 1/3
