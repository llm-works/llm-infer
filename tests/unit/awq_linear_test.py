"""Unit tests for AWQ quantized linear layers."""

import numpy as np
import pytest
import torch

from llm_infer.backends.linear.formats import QuantFormat
from llm_infer.backends.linear.kernels.awq_pytorch import PyTorchAWQBackend
from llm_infer.pipelines.model.layers import AWQLinear

pytestmark = pytest.mark.unit


class TestPyTorchAWQBackendUnpack:
    """Test INT4 unpacking in PyTorch backend."""

    def test_unpack_single_value(self) -> None:
        """Test unpacking a single INT32 with AWQ GEMM interleaved packing.

        AWQ GEMM format uses interleaved packing order: [0, 2, 4, 6, 1, 3, 5, 7]
        This means to store weights [0, 1, 2, 3, 4, 5, 6, 7], the packed nibbles are:
        [0, 2, 4, 6, 1, 3, 5, 7] -> 0x75316420
        """
        backend = PyTorchAWQBackend()

        # AWQ GEMM packed representation of weights [0, 1, 2, 3, 4, 5, 6, 7]
        # Nibbles in bit order: [0, 2, 4, 6, 1, 3, 5, 7]
        # 0x75316420 = 0111_0101_0011_0001_0110_0100_0010_0000
        packed = torch.tensor([[0x75316420]], dtype=torch.int32)

        unpacked = backend._unpack_int4(packed)

        assert unpacked.shape == (1, 8)
        expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int8)
        assert torch.equal(unpacked, expected)

    def test_unpack_max_values(self) -> None:
        """Test unpacking INT32 with max INT4 values (15)."""
        backend = PyTorchAWQBackend()

        # All 0xF = 15 (use numpy to handle signed int32 correctly)
        packed = torch.from_numpy(np.array([[np.int32(-1)]], dtype=np.int32))

        unpacked = backend._unpack_int4(packed)

        assert unpacked.shape == (1, 8)
        expected = torch.tensor([[15, 15, 15, 15, 15, 15, 15, 15]], dtype=torch.int8)
        assert torch.equal(unpacked, expected)

    def test_unpack_preserves_batch_dims(self) -> None:
        """Test that unpacking preserves leading dimensions."""
        backend = PyTorchAWQBackend()

        # Shape: (2, 3, 4) -> should become (2, 3, 32)
        packed = torch.zeros(2, 3, 4, dtype=torch.int32)

        unpacked = backend._unpack_int4(packed)

        assert unpacked.shape == (2, 3, 32)


class TestPyTorchAWQBackendDequantize:
    """Test dequantization in PyTorch backend."""

    def test_dequantize_identity(self) -> None:
        """Test dequantization with identity scale and zero zeros."""
        backend = PyTorchAWQBackend()

        in_features, out_features = 128, 128
        group_size = 128
        num_groups = in_features // group_size
        pack_factor = 8

        # scales = 1.0
        scales = torch.ones(num_groups, out_features, dtype=torch.float16)
        # zeros = 0
        qzeros = torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32)
        # weights = 2 in all positions: 0x22222222
        qweight = torch.full(
            (in_features, out_features // pack_factor), 0x22222222, dtype=torch.int32
        )

        weight = backend._dequantize(qweight, scales, qzeros, group_size)

        # Result is transposed: (out_features, in_features)
        assert weight.shape == (out_features, in_features)
        # All values should be 2.0 (since scale=1, zero=0)
        assert torch.allclose(weight, torch.full_like(weight, 2.0))

    def test_dequantize_with_scale(self) -> None:
        """Test dequantization with non-unity scale."""
        backend = PyTorchAWQBackend()

        in_features, out_features = 128, 128
        group_size = 128
        num_groups = in_features // group_size
        pack_factor = 8

        # scales = 0.5
        scales = torch.full((num_groups, out_features), 0.5, dtype=torch.float16)
        # zeros = 0
        qzeros = torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32)
        # weights = 4 in all positions: 0x44444444
        qweight = torch.full(
            (in_features, out_features // pack_factor), 0x44444444, dtype=torch.int32
        )

        weight = backend._dequantize(qweight, scales, qzeros, group_size)

        # All values should be 2.0 (4 * 0.5)
        assert weight.shape == (out_features, in_features)
        assert torch.allclose(weight, torch.full_like(weight, 2.0))

    def test_dequantize_with_zeros(self) -> None:
        """Test dequantization with non-zero zero points."""
        backend = PyTorchAWQBackend()

        in_features, out_features = 128, 128
        group_size = 128
        num_groups = in_features // group_size
        pack_factor = 8

        # scales = 1.0
        scales = torch.ones(num_groups, out_features, dtype=torch.float16)
        # zeros = 2 (pack 8 x 2 into each int32): 0x22222222
        qzeros = torch.full(
            (num_groups, out_features // pack_factor), 0x22222222, dtype=torch.int32
        )
        # weights = 6: 0x66666666
        qweight = torch.full(
            (in_features, out_features // pack_factor), 0x66666666, dtype=torch.int32
        )

        weight = backend._dequantize(qweight, scales, qzeros, group_size)

        # All values should be 4.0 ((6 - 2) * 1.0)
        assert weight.shape == (out_features, in_features)
        assert torch.allclose(weight, torch.full_like(weight, 4.0))


class TestQuantizedLinearInit:
    """Test QuantizedLinear initialization."""

    def test_awq_creates_correct_buffer_shapes(self) -> None:
        """Test that AWQ init creates buffers with correct shapes."""
        layer = AWQLinear(1024, 4096, group_size=128)

        # qweight: (in_features, out_features // 8)
        assert layer.qweight.shape == (1024, 512)
        assert layer.qweight.dtype == torch.int32

        # qzeros: (num_groups, out_features // 8)
        assert layer.qzeros.shape == (8, 512)  # 1024 // 128 = 8 groups
        assert layer.qzeros.dtype == torch.int32

        # scales: (num_groups, out_features)
        assert layer.scales.shape == (8, 4096)
        assert layer.scales.dtype == torch.float16

    def test_awq_validates_in_features_divisibility(self) -> None:
        """Test that in_features must be divisible by group_size."""
        with pytest.raises(ValueError, match="in_features"):
            AWQLinear(100, 256, group_size=128)  # 100 not divisible by 128

    def test_awq_validates_out_features_divisibility(self) -> None:
        """Test that out_features must be divisible by pack_factor (8)."""
        with pytest.raises(ValueError, match="out_features"):
            AWQLinear(128, 100, group_size=128)  # 100 not divisible by 8

    def test_format_is_set_correctly(self) -> None:
        """Test that format is set correctly for AWQ."""
        layer = AWQLinear(128, 256, group_size=128)
        assert layer.format == QuantFormat.AWQ


class TestQuantizedLinearForward:
    """Test forward pass with PyTorch backend."""

    def test_forward_shape(self) -> None:
        """Test that forward pass produces correct output shape."""
        backend = PyTorchAWQBackend()
        layer = AWQLinear(128, 256, group_size=128, backend=backend)

        x = torch.randn(2, 10, 128, dtype=torch.float16)
        y = layer(x)

        assert y.shape == (2, 10, 256)
        assert y.dtype == torch.float16

    def test_forward_deterministic(self) -> None:
        """Test that forward pass is deterministic."""
        backend = PyTorchAWQBackend()
        layer = AWQLinear(128, 256, group_size=128, backend=backend)

        x = torch.randn(2, 10, 128, dtype=torch.float16)
        y1 = layer(x)
        y2 = layer(x)

        assert torch.equal(y1, y2)
