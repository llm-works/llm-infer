"""Base protocol and types for quantized linear backends.

This module defines the format-agnostic abstractions for quantized linear layers:
- QuantFormat: Enum for supported quantization formats (AWQ, FP8, etc.)
- QuantizedWeights: Union type for format-specific weight containers
- QuantizedLinearBackend: Protocol that all backends must implement
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, Union, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    from .awq import AWQWeights
    from .fp8 import FP8Weights


class QuantFormat(Enum):
    """Supported weight quantization formats.

    Each format has different:
    - Bit width (INT4, FP8, etc.)
    - Scaling granularity (per-group, per-block, per-tensor)
    - Packing scheme
    - Optimal backends
    """

    NONE = auto()  # Full precision (FP16/BF16/FP32)
    AWQ = auto()  # 4-bit with per-group scales (group_size typically 128)
    FP8 = auto()  # FP8 with per-block scales (block_size typically 128)

    @classmethod
    def from_quant_method(cls, method: str | None) -> QuantFormat:
        """Convert HuggingFace quant_method string to QuantFormat.

        Args:
            method: Quantization method from model config (e.g., "awq", "fp8")

        Returns:
            Corresponding QuantFormat, or NONE if unrecognized
        """
        if method is None:
            return cls.NONE
        mapping = {
            "awq": cls.AWQ,
            "fp8": cls.FP8,
        }
        return mapping.get(method.lower(), cls.NONE)


# Union type for all weight containers (forward declaration resolved at runtime)
QuantizedWeights = Union["AWQWeights", "FP8Weights"]


@runtime_checkable
class QuantizedLinearBackend(Protocol):
    """Protocol for quantized linear backends.

    Backends provide the core matmul operation for a specific quantization format.
    Each backend can trade off:
    - Speed (fused CUDA kernels vs pure PyTorch)
    - Dependencies (standalone vs external libraries like vLLM)
    - Hardware support (compute capability requirements)

    Attributes:
        name: Human-readable backend name (e.g., "pytorch", "marlin", "cutlass")
        format: The quantization format this backend supports
    """

    name: str
    format: QuantFormat

    def is_available(self) -> bool:
        """Check if this backend is available.

        Returns:
            True if all dependencies are installed and hardware requirements met.
        """
        ...

    def forward(self, x: Tensor, weights: QuantizedWeights) -> Tensor:
        """Perform quantized matrix multiplication.

        Args:
            x: Input tensor [..., in_features], dtype typically float16/bfloat16
            weights: Format-specific weights container (AWQWeights, FP8Weights, etc.)

        Returns:
            Output tensor [..., out_features], same dtype as input
        """
        ...
