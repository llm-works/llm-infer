"""Marlin CUDA kernel backend for AWQ linear operations.

This backend uses optimized CUDA kernels for fused dequantization + matmul,
achieving near-FP16 speeds with 4-bit quantized weights.

Requires:
- vLLM package with Marlin kernels
- CUDA compute capability >= 8.0 (Ampere or newer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from ..formats.awq import AWQWeights
from ..formats.base import QuantFormat

logger = logging.getLogger(__name__)

# Check for vLLM Marlin ops availability at import time
_VLLM_MARLIN_AVAILABLE = False
_VLLM_MARLIN_ERROR: str | None = None

try:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        awq_to_marlin_zero_points,
        marlin_make_empty_g_idx,
        marlin_make_workspace_new,
        marlin_permute_bias,
        marlin_permute_scales,
    )
    from vllm.scalar_type import scalar_types

    # Verify the specific ops we need are available
    assert hasattr(ops, "awq_marlin_repack"), "awq_marlin_repack not found"
    assert hasattr(ops, "gptq_marlin_gemm"), "gptq_marlin_gemm not found"
    _VLLM_MARLIN_AVAILABLE = True
except (ImportError, AttributeError, AssertionError) as e:
    _VLLM_MARLIN_ERROR = str(e)


@dataclass
class _MarlinWeights:
    """Cached Marlin-format weights for a single AWQ layer.

    Marlin kernels require weights in a specific layout optimized for
    tensor core operations. This dataclass holds the repacked weights
    and associated tensors needed for the GEMM kernel.
    """

    qweight: Tensor  # Repacked weights in Marlin format
    scales: Tensor  # Permuted scales
    qzeros: Tensor  # Converted zero points
    bias: Tensor | None  # Permuted bias (or None)
    workspace: Tensor  # Kernel workspace buffer
    g_idx: Tensor  # Group index (empty for AWQ)
    g_idx_sort: Tensor  # Sort indices (empty for AWQ)
    in_features: int
    out_features: int


class MarlinAWQBackend:
    """Marlin CUDA kernel backend for fast AWQ inference.

    This backend provides 10-15x speedup over pure PyTorch by using
    fused dequantization + matmul CUDA kernels from vLLM.

    The Marlin kernel requires:
    - Ampere+ GPU (compute capability >= 8.0)
    - Weights repacked to Marlin format (done on first use, cached)

    Falls back to PyTorch backend if unavailable.
    """

    name: str = "marlin"
    format: QuantFormat = QuantFormat.AWQ

    def __init__(self) -> None:
        """Initialize Marlin backend."""
        # Cache repacked weights by tensor data_ptr
        # This ensures we only repack once per layer
        self._cache: dict[int, _MarlinWeights] = {}

    def is_available(self) -> bool:
        """Check if Marlin backend is available.

        Requires:
        - vLLM package with Marlin ops
        - CUDA available
        - Compute capability >= 8.0
        """
        if not _VLLM_MARLIN_AVAILABLE:
            logger.debug(f"vLLM Marlin ops not available: {_VLLM_MARLIN_ERROR}")
            return False

        if not torch.cuda.is_available():
            logger.debug("CUDA not available")
            return False

        # Check compute capability
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            logger.debug(
                f"Compute capability {capability[0]}.{capability[1]} < 8.0, "
                "Marlin requires Ampere or newer"
            )
            return False

        return True

    def _repack_weights_to_marlin(
        self,
        qweight: Tensor,
        scales: Tensor,
        qzeros: Tensor,
        bias: Tensor | None,
        group_size: int,
        in_features: int,
        out_features: int,
    ) -> _MarlinWeights:
        """Repack AWQ weights to Marlin format."""
        marlin_qweight = ops.awq_marlin_repack(
            qweight, size_k=in_features, size_n=out_features, num_bits=4
        )
        marlin_scales = marlin_permute_scales(
            scales, size_k=in_features, size_n=out_features, group_size=group_size
        )
        num_groups = in_features // group_size
        marlin_zp = awq_to_marlin_zero_points(
            qzeros, size_k=num_groups, size_n=out_features, num_bits=4
        )
        # Permute bias to match Marlin output order
        marlin_bias = marlin_permute_bias(bias) if bias is not None else None

        device = qweight.device
        workspace = marlin_make_workspace_new(device)
        g_idx = marlin_make_empty_g_idx(device)

        return _MarlinWeights(
            qweight=marlin_qweight,
            scales=marlin_scales,
            qzeros=marlin_zp,
            bias=marlin_bias,
            workspace=workspace,
            g_idx=g_idx,
            g_idx_sort=g_idx,
            in_features=in_features,
            out_features=out_features,
        )

    def _get_or_create_marlin_weights(self, weights: AWQWeights) -> _MarlinWeights:
        """Get cached Marlin weights or create and cache them.

        Weight repacking is expensive (~10ms per layer for 7B model),
        so we cache the repacked weights by the qweight tensor's data_ptr.
        """
        cache_key = weights.qweight.data_ptr()
        if cache_key in self._cache:
            return self._cache[cache_key]

        in_features = weights.in_features
        out_features = weights.out_features

        logger.debug(
            f"Repacking weights to Marlin format: "
            f"{in_features}x{out_features}, group_size={weights.group_size}"
        )

        cached = self._repack_weights_to_marlin(
            weights.qweight,
            weights.scales,
            weights.qzeros,
            weights.bias,
            weights.group_size,
            in_features,
            out_features,
        )
        self._cache[cache_key] = cached
        return cached

    def _call_marlin_gemm(self, x: Tensor, mw) -> Tensor:
        """Call Marlin GEMM kernel with prepared weights."""
        return ops.gptq_marlin_gemm(
            x,
            None,
            mw.qweight,
            mw.bias,
            mw.scales,
            None,
            None,
            mw.qzeros,
            mw.g_idx,
            mw.g_idx_sort,
            mw.workspace,
            scalar_types.uint4,
            size_m=x.shape[0],
            size_n=mw.out_features,
            size_k=mw.in_features,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=False,
            is_zp_float=False,
        )

    def forward(self, x: Tensor, weights: AWQWeights) -> Tensor:
        """Perform AWQ quantized matrix multiplication using Marlin kernel."""
        mw = self._get_or_create_marlin_weights(weights)

        # Marlin kernel requires float16 or bfloat16 input
        if x.dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.float16)

        # Reshape for GEMM: [batch, seq, in] -> [batch*seq, in]
        orig_shape = x.shape
        output = self._call_marlin_gemm(x.reshape(-1, x.shape[-1]), mw)

        # Reshape back to original batch dimensions
        return output.reshape(orig_shape[:-1] + (mw.out_features,))
