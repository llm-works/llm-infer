"""Backend abstraction layer for inference components.

This module provides pluggable backends at three levels:
- linear: Kernel-level backends for quantized matmul (e.g., pytorch, marlin)
- model: Model-level backends (e.g., native TransformerModel, gptqmodel)
- engine: Full engine backends (e.g., native, vllm)

Higher levels override lower levels in the hierarchy.
"""

from .linear import QuantizedLinearBackend, get_linear_backend

__all__ = [
    "get_linear_backend",
    "QuantizedLinearBackend",
]
