"""Pure PyTorch backend for AWQ linear operations.

This backend performs dequantization using standard PyTorch operations.
It's slower than optimized CUDA kernels but requires no external dependencies
and works on any hardware that supports PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ..formats.awq import AWQWeights
from ..formats.base import QuantFormat

PACK_FACTOR = 8  # 8 x INT4 values per INT32


class PyTorchAWQBackend:
    """Pure PyTorch implementation of AWQ dequantization.

    This is the fallback backend that always works. It performs:
    1. INT4 unpacking from INT32 (with AWQ GEMM reordering)
    2. Dequantization: weight = (qweight - zeros) * scales
    3. Standard F.linear matrix multiplication

    These operations are separate, not fused, so performance is limited
    by memory bandwidth for the intermediate tensors.
    """

    name: str = "pytorch"
    format: QuantFormat = QuantFormat.AWQ

    def is_available(self) -> bool:
        """Always available - uses only PyTorch."""
        return True

    def forward(self, x: Tensor, weights: AWQWeights) -> Tensor:
        """Perform AWQ quantized matrix multiplication.

        Args:
            x: Input tensor [..., in_features]
            weights: AWQ weights container

        Returns:
            Output tensor [..., out_features]
        """
        # Dequantize weights
        weight = self._dequantize(
            weights.qweight, weights.scales, weights.qzeros, weights.group_size
        )

        # Cast input to match weight dtype
        x = x.to(weight.dtype)

        return F.linear(x, weight, weights.bias)

    def _dequantize(
        self,
        qweight: Tensor,
        scales: Tensor,
        qzeros: Tensor,
        group_size: int,
    ) -> Tensor:
        """Dequantize weights from INT4 to FP16.

        Returns:
            Dequantized weight tensor, shape (out_features, in_features)
            Note: Transposed for use with F.linear
        """
        # Unpack qweight: (in_features, out_features // 8) -> (in_features, out_features)
        weight_int = self._unpack_int4(qweight)

        # Unpack qzeros: (num_groups, out_features // 8) -> (num_groups, out_features)
        zeros_int = self._unpack_int4(qzeros)

        # Expand zeros and scales to match weight dimensions
        # Each group covers group_size rows of weights
        zeros_expanded = zeros_int.repeat_interleave(group_size, dim=0)
        scales_expanded = scales.repeat_interleave(group_size, dim=0)

        # Dequantize: weight = (qweight - zeros) * scales
        weight = (
            weight_int.to(scales_expanded.dtype)
            - zeros_expanded.to(scales_expanded.dtype)
        ) * scales_expanded

        # Transpose for F.linear: (in_features, out_features) -> (out_features, in_features)
        return weight.t()

    def _unpack_int4(self, packed: Tensor) -> Tensor:
        """Unpack INT32 tensor to INT4 values.

        Each INT32 contains 8 x INT4 values packed in AWQ GEMM interleaved order.
        AWQ GEMM format packs weights using order [0, 2, 4, 6, 1, 3, 5, 7] to
        enable efficient memory access patterns in GEMM kernels.

        Args:
            packed: Tensor of packed INT32 values, shape (..., N)

        Returns:
            Unpacked tensor, shape (..., N * 8) with dtype int8
        """
        *prefix, packed_dim = packed.shape
        unpacked_dim = packed_dim * PACK_FACTOR

        # Expand for unpacking: (..., N, 1)
        packed = packed.unsqueeze(-1)

        # Shift amounts for extracting each 4-bit nibble
        shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)

        # Extract nibbles: (..., N, 8)
        unpacked = (packed >> shifts) & 0xF

        # AWQ GEMM uses interleaved packing [0, 2, 4, 6, 1, 3, 5, 7]
        # Inverse to restore sequential order: [0, 4, 1, 5, 2, 6, 3, 7]
        reorder_idx = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=packed.device)
        unpacked = unpacked.index_select(-1, reorder_idx)

        # Reshape to (..., N * 8)
        unpacked = unpacked.view(*prefix, unpacked_dim)

        return unpacked.to(torch.int8)
