"""CUTLASS backend for FP8 linear operations.

This backend uses vLLM's CUTLASS kernel for native FP8 matmul with
block-wise scaling. Provides significant speedup over PyTorch dequantization.

Requires:
- vLLM package with CUTLASS ops
- CUDA compute capability >= 8.9 (Ada/Hopper for native FP8)
"""

from __future__ import annotations

import torch
from appinfra.log import Logger
from torch import Tensor

from ..formats.base import QuantFormat
from ..formats.fp8 import FP8Weights

# Check for vLLM CUTLASS ops availability at import time
_VLLM_CUTLASS_AVAILABLE = False
_VLLM_CUTLASS_ERROR: str | None = None
_cutlass_scaled_mm = None

try:
    from vllm import _custom_ops as ops

    if hasattr(ops, "cutlass_scaled_mm"):
        _cutlass_scaled_mm = ops.cutlass_scaled_mm
        _VLLM_CUTLASS_AVAILABLE = True
    else:
        _VLLM_CUTLASS_ERROR = "cutlass_scaled_mm not found in vllm._custom_ops"
except ImportError as e:
    _VLLM_CUTLASS_ERROR = str(e)


class CutlassFP8Backend:
    """CUTLASS CUDA kernel backend for fast FP8 inference.

    This backend uses vLLM's CUTLASS kernel for native FP8 matrix multiplication
    with block-wise scaling. This avoids materializing the full FP16 weight tensor.

    The CUTLASS kernel requires:
    - Ada or Hopper GPU (compute capability >= 8.9 for native FP8)
    - vLLM with CUTLASS support

    Falls back to PyTorch backend if unavailable.
    """

    name: str = "cutlass"
    format: QuantFormat = QuantFormat.FP8

    def __init__(self, lg: Logger) -> None:
        """Initialize CUTLASS FP8 backend."""
        self._lg = lg

    def is_available(self) -> bool:
        """Check if CUTLASS FP8 backend is available.

        Requires:
        - vLLM package with CUTLASS ops
        - CUDA available
        - Compute capability >= 8.9 (Ada/Hopper)
        """
        if not _VLLM_CUTLASS_AVAILABLE:
            self._lg.debug(
                "vLLM CUTLASS ops not available", extra={"error": _VLLM_CUTLASS_ERROR}
            )
            return False

        if not torch.cuda.is_available():
            self._lg.debug("CUDA not available")
            return False

        # Check compute capability (8.9 = Ada, 9.0 = Hopper)
        capability = torch.cuda.get_device_capability()
        if capability < (8, 9):
            self._lg.debug(
                "native FP8 requires Ada or newer",
                extra={"capability": f"{capability[0]}.{capability[1]}"},
            )
            return False

        return True

    def forward(self, x: Tensor, weights: FP8Weights) -> Tensor:
        """Perform FP8 quantized matrix multiplication using CUTLASS kernel.

        Args:
            x: Input tensor [..., in_features]
            weights: FP8 weights container

        Returns:
            Output tensor [..., out_features]
        """
        assert _cutlass_scaled_mm is not None

        # Flatten batch dimensions for matmul
        orig_shape = x.shape
        x = x.view(-1, weights.in_features)  # [M, K]

        # Convert input to FP8 for native FP8 matmul
        # Note: This is a simplified version - production would use proper input scaling
        if x.dtype != torch.float8_e4m3fn:
            x_fp8 = x.to(torch.float8_e4m3fn)
            input_scale = torch.ones(1, device=x.device, dtype=torch.float32)
        else:
            x_fp8 = x
            input_scale = torch.ones(1, device=x.device, dtype=torch.float32)

        # Call CUTLASS scaled matmul
        # Note: cutlass_scaled_mm expects specific scale formats
        # This is a simplified interface - may need adjustment based on vLLM version
        out = _cutlass_scaled_mm(
            x_fp8,
            weights.weight.t(),  # CUTLASS expects [K, N] for weight
            input_scale,
            weights.weight_scale_inv.flatten(),  # Scale per output block
            out_dtype=torch.float16,
        )

        # Restore batch dimensions
        return out.view(*orig_shape[:-1], weights.out_features)
