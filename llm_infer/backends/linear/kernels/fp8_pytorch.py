"""Pure PyTorch backend for FP8 linear operations.

This backend performs block-wise dequantization using standard PyTorch operations.
It's slower than native FP8 CUDA kernels but requires no external dependencies
beyond PyTorch with FP8 support.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ..formats.base import QuantFormat
from ..formats.fp8 import FP8Weights


class PyTorchFP8Backend:
    """Pure PyTorch implementation of FP8 dequantization.

    This is the fallback backend for FP8 that always works. It performs:
    1. Block-wise dequantization: weight_fp16 = weight_fp8 * scale
    2. Standard F.linear matrix multiplication

    These operations are separate, not fused, so performance is limited
    by memory bandwidth for the intermediate FP16 weights.
    """

    name: str = "pytorch"
    format: QuantFormat = QuantFormat.FP8

    def is_available(self) -> bool:
        """Available if PyTorch supports FP8 dtype."""
        return hasattr(torch, "float8_e4m3fn")

    def forward(self, x: Tensor, weights: FP8Weights) -> Tensor:
        """Perform FP8 quantized matrix multiplication.

        Args:
            x: Input tensor [..., in_features]
            weights: FP8 weights container

        Returns:
            Output tensor [..., out_features]
        """
        # Flatten batch dimensions for matmul
        orig_shape = x.shape
        x = x.view(-1, weights.in_features)  # [M, K]

        # Dequantize weights block-by-block
        weight_dequant = self._dequantize(weights)

        # Standard FP16 matmul
        out = F.linear(x, weight_dequant)

        # Restore batch dimensions
        return out.view(*orig_shape[:-1], weights.out_features)

    def _dequantize(self, weights: FP8Weights) -> Tensor:
        """Dequantize FP8 weights to FP16.

        Applies per-block scaling to convert FP8 weights to FP16.

        Returns:
            Dequantized weight tensor [out_features, in_features] in FP16
        """
        out_features = weights.out_features
        in_features = weights.in_features
        block_size = weights.block_size

        out_blocks = out_features // block_size
        in_blocks = in_features // block_size

        # Reshape weight to blocks: [out_blocks, block, in_blocks, block]
        weight_blocks = weights.weight.view(
            out_blocks, block_size, in_blocks, block_size
        )

        # scale_inv: [out_blocks, in_blocks] -> [out_blocks, 1, in_blocks, 1]
        # weight_scale_inv IS the inverse scale (multiply by it to dequantize)
        scales = weights.weight_scale_inv.view(out_blocks, 1, in_blocks, 1)

        # Dequantize: multiply each block by its scale
        weight_dequant = weight_blocks.to(torch.float16) * scales.to(torch.float16)

        # Reshape back: [out_features, in_features]
        return weight_dequant.view(out_features, in_features)
