"""Unit tests for RMSNorm layer."""

import pytest
import torch

from llm_infer.pipelines.model.layers import RMSNorm

pytestmark = pytest.mark.unit


class TestRMSNormInit:
    """Test RMSNorm initialization."""

    def test_creates_weight_parameter(self) -> None:
        """Test that weight parameter is created."""
        norm = RMSNorm(hidden_size=64)
        assert hasattr(norm, "weight")
        assert norm.weight.shape == (64,)

    def test_weight_initialized_to_ones(self) -> None:
        """Test that weight is initialized to ones."""
        norm = RMSNorm(hidden_size=64)
        assert torch.allclose(norm.weight, torch.ones(64))

    def test_default_eps(self) -> None:
        """Test default epsilon value."""
        norm = RMSNorm(hidden_size=64)
        assert norm.eps == 1e-5

    def test_custom_eps(self) -> None:
        """Test custom epsilon value."""
        norm = RMSNorm(hidden_size=64, eps=1e-6)
        assert norm.eps == 1e-6


class TestRMSNormForward:
    """Test RMSNorm forward pass."""

    def test_output_shape_matches_input(self) -> None:
        """Test that output shape matches input."""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape

    def test_preserves_dtype(self) -> None:
        """Test that output dtype matches input dtype."""
        norm = RMSNorm(hidden_size=64)

        x_fp16 = torch.randn(2, 10, 64, dtype=torch.float16)
        y_fp16 = norm(x_fp16)
        assert y_fp16.dtype == torch.float16

        x_fp32 = torch.randn(2, 10, 64, dtype=torch.float32)
        y_fp32 = norm(x_fp32)
        assert y_fp32.dtype == torch.float32

    def test_normalizes_vectors(self) -> None:
        """Test that output is normalized (RMS close to 1)."""
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 10, 64)
        y = norm(x)

        # After RMSNorm, the RMS of each vector should be close to 1
        # (scaled by weight which is 1)
        rms = torch.sqrt(torch.mean(y**2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_zero_input_does_not_crash(self) -> None:
        """Test that zero input doesn't cause NaN (eps prevents div by zero)."""
        norm = RMSNorm(hidden_size=64)
        x = torch.zeros(2, 10, 64)
        y = norm(x)
        assert not torch.isnan(y).any()

    def test_weight_scales_output(self) -> None:
        """Test that weight parameter scales output."""
        norm = RMSNorm(hidden_size=4)
        norm.weight.data = torch.tensor([2.0, 2.0, 2.0, 2.0])

        # Use ones for predictable RMS
        x = torch.ones(1, 1, 4)
        y = norm(x)

        # RMS of [1,1,1,1] = 1, so output = weight * 1 = [2,2,2,2]
        assert torch.allclose(y, torch.tensor([[[2.0, 2.0, 2.0, 2.0]]]))

    def test_batch_independence(self) -> None:
        """Test that batches are normalized independently."""
        norm = RMSNorm(hidden_size=64)
        x1 = torch.randn(1, 10, 64)
        x2 = torch.randn(1, 10, 64) * 10  # Different scale

        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = norm(x_batch)

        y1_single = norm(x1)
        y2_single = norm(x2)

        assert torch.allclose(y_batch[0], y1_single[0], atol=1e-5)
        assert torch.allclose(y_batch[1], y2_single[0], atol=1e-5)
