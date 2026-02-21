"""Linear backend selection and management.

This module provides a unified interface for quantized linear operations
supporting multiple formats (AWQ, FP8) with priority-based backend selection.

Quick start:
    >>> from appinfra.log import create_lg
    >>> from llm_infer.engines.native.backends.linear import BackendRegistry, QuantFormat
    >>> lg = create_lg(__name__, "info")
    >>> registry = BackendRegistry(lg)
    >>> backend = registry.get(QuantFormat.AWQ)  # Auto-selects best available
    >>> output = backend.forward(x, awq_weights)

Get specific backend:
    >>> backend = registry.get(QuantFormat.AWQ, preference="marlin")
"""

from __future__ import annotations

# Core types and protocols
from .formats import AWQWeights, FP8Weights, QuantFormat, QuantizedLinearBackend

# Registry class
from .registry import BackendRegistry

__all__ = [
    # Format types
    "QuantFormat",
    "QuantizedLinearBackend",
    "AWQWeights",
    "FP8Weights",
    # Registry
    "BackendRegistry",
]
