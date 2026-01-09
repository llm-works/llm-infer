"""Unit tests for QuantizedLinear layer variants."""

import pytest
import torch

from llm_infer.backends.linear.formats import QuantFormat
from llm_infer.backends.linear.kernels.awq_pytorch import PyTorchAWQBackend
from llm_infer.pipelines.model.layers import (
    AWQLinear,
    Fp8Linear,
    QuantizedLinear,
)

pytestmark = pytest.mark.unit


class TestAWQLinearFactory:
    """Test AWQLinear factory function."""

    def test_creates_awq_format(self) -> None:
        """Test AWQLinear creates QuantizedLinear with AWQ format."""
        layer = AWQLinear(in_features=128, out_features=256)
        assert isinstance(layer, QuantizedLinear)
        assert layer.format == QuantFormat.AWQ

    def test_passes_group_size(self) -> None:
        """Test AWQLinear passes group_size."""
        layer = AWQLinear(in_features=256, out_features=256, group_size=64)
        assert layer.group_size == 64

    def test_passes_bias(self) -> None:
        """Test AWQLinear passes bias parameter."""
        layer = AWQLinear(in_features=128, out_features=256, bias=True)
        assert layer.bias is not None


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 not supported in this PyTorch version",
)
class TestFp8LinearFactory:
    """Test Fp8Linear factory function."""

    def test_creates_fp8_format(self) -> None:
        """Test Fp8Linear creates QuantizedLinear with FP8 format."""
        layer = Fp8Linear(in_features=128, out_features=256)
        assert isinstance(layer, QuantizedLinear)
        assert layer.format == QuantFormat.FP8

    def test_passes_block_size(self) -> None:
        """Test Fp8Linear passes block_size."""
        layer = Fp8Linear(in_features=256, out_features=256, block_size=64)
        assert layer.block_size == 64


class TestQuantizedLinearExtraRepr:
    """Test QuantizedLinear extra_repr method."""

    def test_awq_extra_repr(self) -> None:
        """Test AWQ format extra_repr."""
        layer = AWQLinear(in_features=128, out_features=256, group_size=128)
        repr_str = layer.extra_repr()

        assert "in_features=128" in repr_str
        assert "out_features=256" in repr_str
        assert "format=AWQ" in repr_str
        assert "group_size=128" in repr_str
        assert "backend=" in repr_str

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e4m3fn"),
        reason="FP8 not supported in this PyTorch version",
    )
    def test_fp8_extra_repr(self) -> None:
        """Test FP8 format extra_repr."""
        layer = Fp8Linear(in_features=128, out_features=256, block_size=128)
        repr_str = layer.extra_repr()

        assert "in_features=128" in repr_str
        assert "out_features=256" in repr_str
        assert "format=FP8" in repr_str
        assert "block_size=128" in repr_str


class TestQuantizedLinearBackendSetter:
    """Test QuantizedLinear backend setter."""

    def test_set_backend(self) -> None:
        """Test setting backend via property."""
        layer = AWQLinear(in_features=128, out_features=256)
        backend = PyTorchAWQBackend()
        layer.backend = backend
        assert layer._backend is backend

    def test_lazy_backend_init(self) -> None:
        """Test backend is lazily initialized."""
        layer = AWQLinear(in_features=128, out_features=256)
        assert layer._backend is None
        # Accessing backend property should initialize it
        _ = layer.backend
        assert layer._backend is not None


class TestQuantizedLinearValidation:
    """Test QuantizedLinear validation."""

    def test_invalid_format_raises(self) -> None:
        """Test unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            QuantizedLinear(
                in_features=128,
                out_features=256,
                format="invalid",  # type: ignore
            )

    def test_awq_invalid_group_size(self) -> None:
        """Test AWQ with invalid group_size raises."""
        with pytest.raises(ValueError, match="must be divisible by group_size"):
            AWQLinear(in_features=100, out_features=256, group_size=128)

    def test_awq_invalid_out_features(self) -> None:
        """Test AWQ with out_features not divisible by pack_factor raises."""
        with pytest.raises(ValueError, match="must be divisible by pack_factor"):
            AWQLinear(in_features=128, out_features=100, group_size=128)
