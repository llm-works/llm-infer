"""Unit tests for FP8 PyTorch backend."""

import pytest
import torch

from llm_infer.backends.linear.formats import QuantFormat
from llm_infer.backends.linear.formats.fp8 import FP8Weights
from llm_infer.backends.linear.kernels.fp8_pytorch import PyTorchFP8Backend

pytestmark = pytest.mark.unit


class TestPyTorchFP8BackendAvailability:
    """Test backend availability."""

    def test_is_available(self) -> None:
        """Test that backend is available when PyTorch has FP8 support."""
        backend = PyTorchFP8Backend()
        # Should be True for PyTorch >= 2.1 with CUDA
        expected = hasattr(torch, "float8_e4m3fn")
        assert backend.is_available() == expected

    def test_name_and_format(self) -> None:
        """Test backend name and format."""
        backend = PyTorchFP8Backend()
        assert backend.name == "pytorch"
        assert backend.format == QuantFormat.FP8


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 not supported in this PyTorch version",
)
class TestPyTorchFP8BackendDequantize:
    """Test FP8 dequantization."""

    def test_dequantize_shape(self) -> None:
        """Test that dequantize produces correct shape."""
        backend = PyTorchFP8Backend()

        out_features, in_features = 256, 128
        block_size = 128

        weight = torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        weight_scale_inv = torch.ones(2, 1, dtype=torch.float16)  # 256/128, 128/128

        weights = FP8Weights(
            weight=weight,
            weight_scale_inv=weight_scale_inv,
            block_size=block_size,
        )

        dequant = backend._dequantize(weights)

        assert dequant.shape == (out_features, in_features)
        assert dequant.dtype == torch.float16

    def test_dequantize_applies_scale(self) -> None:
        """Test that dequantization applies scale correctly."""
        backend = PyTorchFP8Backend()

        out_features, in_features = 128, 128
        block_size = 128

        # Create FP8 weights with value 1.0
        weight_fp32 = torch.ones(out_features, in_features)
        weight = weight_fp32.to(torch.float8_e4m3fn)

        # Scale of 2.0 should double the values
        weight_scale_inv = torch.full((1, 1), 2.0, dtype=torch.float16)

        weights = FP8Weights(
            weight=weight,
            weight_scale_inv=weight_scale_inv,
            block_size=block_size,
        )

        dequant = backend._dequantize(weights)

        # All values should be approximately 2.0
        assert torch.allclose(dequant, torch.full_like(dequant, 2.0), atol=0.1)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 not supported in this PyTorch version",
)
class TestPyTorchFP8BackendForward:
    """Test FP8 forward pass."""

    def test_forward_shape(self) -> None:
        """Test that forward produces correct output shape."""
        backend = PyTorchFP8Backend()

        out_features, in_features = 256, 128
        block_size = 128

        weight = torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        weight_scale_inv = torch.ones(2, 1, dtype=torch.float16)

        weights = FP8Weights(
            weight=weight,
            weight_scale_inv=weight_scale_inv,
            block_size=block_size,
        )

        x = torch.randn(2, 10, in_features, dtype=torch.float16)
        y = backend.forward(x, weights)

        assert y.shape == (2, 10, out_features)
        assert y.dtype == torch.float16

    def test_forward_deterministic(self) -> None:
        """Test that forward is deterministic."""
        backend = PyTorchFP8Backend()

        out_features, in_features = 128, 128
        block_size = 128

        weight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
        weight_scale_inv = torch.ones(1, 1, dtype=torch.float16)

        weights = FP8Weights(
            weight=weight,
            weight_scale_inv=weight_scale_inv,
            block_size=block_size,
        )

        x = torch.randn(2, 10, in_features, dtype=torch.float16)
        y1 = backend.forward(x, weights)
        y2 = backend.forward(x, weights)

        assert torch.equal(y1, y2)
