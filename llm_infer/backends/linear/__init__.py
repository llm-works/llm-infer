"""Linear backend selection and management.

This module provides a unified interface for quantized linear operations
supporting multiple formats (AWQ, FP8) with priority-based backend selection.

Quick start:
    >>> from llm_infer.backends.linear import get_backend, QuantFormat
    >>> backend = get_backend(QuantFormat.AWQ)  # Auto-selects best available
    >>> output = backend.forward(x, awq_weights)

Backward compatibility:
    >>> from llm_infer.backends.linear import get_linear_backend
    >>> backend = get_linear_backend("marlin")  # AWQ-specific, legacy API
"""

from __future__ import annotations

# Core types and protocols
from .formats import AWQWeights, FP8Weights, QuantFormat, QuantizedLinearBackend

# Registry functions
from .registry import (
    get_available_backends,
    get_backend,
    get_linear_backend,
    register_backend,
)

__all__ = [
    # Format types
    "QuantFormat",
    "QuantizedLinearBackend",
    "AWQWeights",
    "FP8Weights",
    # Registry functions
    "get_backend",
    "get_linear_backend",  # Backward compatibility
    "get_available_backends",
    "register_backend",
]
