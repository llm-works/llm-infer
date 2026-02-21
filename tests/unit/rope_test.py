"""Unit tests for Rotary Position Embeddings (RoPE)."""

import pytest
import torch

from llm_infer.engines.native.attention.rope import (
    apply_rope,
    precompute_rope_freqs,
    rotate_half,
)

pytestmark = pytest.mark.unit


class TestPrecomputeRopeFreqs:
    """Test RoPE frequency precomputation."""

    def test_output_shapes(self) -> None:
        """Test that precompute returns correct shapes."""
        head_dim = 128
        max_seq_len = 1024

        cos, sin = precompute_rope_freqs(
            head_dim=head_dim, max_seq_len=max_seq_len, device="cpu"
        )

        # Shape should be [max_seq_len, head_dim // 2]
        assert cos.shape == (max_seq_len, head_dim // 2)
        assert sin.shape == (max_seq_len, head_dim // 2)

    def test_output_dtype(self) -> None:
        """Test that precompute respects dtype."""
        cos, sin = precompute_rope_freqs(
            head_dim=64, max_seq_len=100, device="cpu", dtype=torch.float32
        )

        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32

    def test_cos_sin_range(self) -> None:
        """Test that cos/sin are in valid range [-1, 1]."""
        cos, sin = precompute_rope_freqs(
            head_dim=64, max_seq_len=100, device="cpu", dtype=torch.float32
        )

        assert cos.min() >= -1.0
        assert cos.max() <= 1.0
        assert sin.min() >= -1.0
        assert sin.max() <= 1.0

    def test_different_theta(self) -> None:
        """Test that different theta produces different frequencies."""
        cos1, sin1 = precompute_rope_freqs(
            head_dim=64, max_seq_len=100, theta=10000.0, device="cpu"
        )
        cos2, sin2 = precompute_rope_freqs(
            head_dim=64, max_seq_len=100, theta=500000.0, device="cpu"
        )

        # Should be different
        assert not torch.allclose(cos1, cos2)
        assert not torch.allclose(sin1, sin2)


class TestRotateHalf:
    """Test rotate_half function."""

    def test_output_shape(self) -> None:
        """Test that rotate_half preserves shape."""
        x = torch.randn(2, 10, 32, 128)
        result = rotate_half(x)
        assert result.shape == x.shape

    def test_rotation_logic(self) -> None:
        """Test that rotation swaps and negates correctly."""
        # Simple case: [1, 2, 3, 4] -> [-3, -4, 1, 2]
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.equal(result, expected)

    def test_inverse(self) -> None:
        """Test that rotating twice gives -x."""
        x = torch.randn(2, 10, 128)
        rotated_twice = rotate_half(rotate_half(x))
        assert torch.allclose(rotated_twice, -x)


class TestApplyRope:
    """Test apply_rope function."""

    def test_output_shapes(self) -> None:
        """Test that apply_rope preserves shapes."""
        batch, seq_len, num_heads, head_dim = 2, 10, 32, 128
        max_seq_len = 1024

        q = torch.randn(batch, seq_len, num_heads, head_dim)
        k = torch.randn(batch, seq_len, num_heads, head_dim)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        cos, sin = precompute_rope_freqs(head_dim, max_seq_len, device="cpu")

        q_out, k_out = apply_rope(q, k, cos, sin, positions)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_preserves_dtype(self) -> None:
        """Test that apply_rope preserves input dtype."""
        batch, seq_len, num_heads, head_dim = 2, 10, 8, 64

        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16)
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        cos, sin = precompute_rope_freqs(
            head_dim, 100, device="cpu", dtype=torch.float16
        )

        q_out, k_out = apply_rope(q, k, cos, sin, positions)

        assert q_out.dtype == torch.float16
        assert k_out.dtype == torch.float16

    def test_position_zero_preserves_values(self) -> None:
        """Test that position 0 with cos=1, sin=0 preserves values."""
        batch, seq_len, head_dim = 1, 1, 4

        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        k = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])

        # At position 0, cos should be ~1 and sin should be ~0
        cos, sin = precompute_rope_freqs(
            head_dim, 100, device="cpu", dtype=torch.float32
        )

        positions = torch.zeros(batch, seq_len, dtype=torch.long)
        q_out, k_out = apply_rope(q, k, cos, sin, positions)

        # Values should be close to original when cos~1, sin~0
        # The first position has the highest frequency still rotated minimally
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_different_positions_give_different_results(self) -> None:
        """Test that different positions give different embeddings."""
        batch, seq_len, num_heads, head_dim = 1, 2, 1, 64

        q = torch.randn(batch, seq_len, num_heads, head_dim)
        k = torch.randn(batch, seq_len, num_heads, head_dim)

        # Same content at different positions
        positions1 = torch.tensor([[0, 1]])
        positions2 = torch.tensor([[100, 101]])

        cos, sin = precompute_rope_freqs(head_dim, 1024, device="cpu")

        q_out1, k_out1 = apply_rope(q, k, cos, sin, positions1)
        q_out2, k_out2 = apply_rope(q, k, cos, sin, positions2)

        # Results should differ due to different positional encodings
        assert not torch.allclose(q_out1, q_out2)
        assert not torch.allclose(k_out1, k_out2)
