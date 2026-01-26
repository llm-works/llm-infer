"""Engine implementations.

This module provides different inference engine backends:
- NativeEngine: Custom implementation for learning/reference (default)
- VLLMEngine: Production-grade vLLM-backed engine (optional)
- OllamaEngine: Connects to running Ollama server (optional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from appinfra.log import Logger

if TYPE_CHECKING:
    from ..engine import InferenceEngine
    from .ollama import OllamaEngine
    from .vllm import VLLMEngine

__all__ = ["create_engine", "VLLMEngine", "OllamaEngine"]


def create_engine(
    lg: Logger,
    engine_type: str,
    config: dict,
) -> InferenceEngine | VLLMEngine | OllamaEngine:
    """Create an inference engine based on type.

    Args:
        lg: Logger instance.
        engine_type: Engine type ("native", "vllm", or "ollama")
        config: Configuration dictionary

    Returns:
        Initialized engine instance

    Raises:
        ValueError: If engine_type is unknown
        ImportError: If vllm engine requested but vllm not installed
    """
    if engine_type == "vllm":
        from .vllm import VLLMEngine

        return VLLMEngine.from_config(lg, config)
    elif engine_type == "ollama":
        from ...serving.dispatch.config import OllamaConfig
        from .ollama import OllamaEngine

        ollama_cfg = OllamaConfig.from_dict(
            config.get("ollama", {}),
            model=config.get("model", {}).get("name", ""),
        )
        return OllamaEngine(lg, ollama_cfg)
    elif engine_type == "native":
        from .native import create_native_engine

        return create_native_engine(lg, config)
    else:
        raise ValueError(
            f"Unknown engine type: {engine_type}. Use 'native', 'vllm', or 'ollama'."
        )
