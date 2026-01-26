"""Engine implementations.

This module provides different inference engine backends:
- NativeEngine: Custom implementation for learning/reference (default)
- VLLMEngine: Production-grade vLLM-backed engine (optional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from appinfra.log import Logger

if TYPE_CHECKING:
    from ..engine import InferenceEngine
    from .vllm_engine import VLLMEngine

__all__ = ["create_engine", "VLLMEngine"]


def create_engine(
    lg: Logger,
    engine_type: str,
    config: dict,
) -> InferenceEngine | VLLMEngine:
    """Create an inference engine based on type.

    Args:
        lg: Logger instance.
        engine_type: Engine type ("native" or "vllm")
        config: Configuration dictionary

    Returns:
        Initialized engine instance

    Raises:
        ValueError: If engine_type is unknown
        ImportError: If vllm engine requested but vllm not installed
    """
    if engine_type == "vllm":
        from .vllm_engine import VLLMEngine

        return VLLMEngine.from_config(lg, config)
    elif engine_type == "native":
        from .native import create_native_engine

        return create_native_engine(lg, config)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}. Use 'native' or 'vllm'.")
