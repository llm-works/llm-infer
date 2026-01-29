"""Backend implementations and registry.

This module provides the backend registry and factory for creating
backends from configuration.
"""

from __future__ import annotations

from typing import Any

from llm_infer.client.backends.base import Backend
from llm_infer.client.backends.openai import OpenAICompatibleBackend

# Registry of backend types
_BACKEND_REGISTRY: dict[str, type[Backend]] = {
    "openai_compatible": OpenAICompatibleBackend,
    "openai": OpenAICompatibleBackend,
}


def register_backend(name: str, backend_class: type[Backend]) -> None:
    """Register a backend type.

    Args:
        name: Name to register the backend under.
        backend_class: Backend class to register.
    """
    _BACKEND_REGISTRY[name] = backend_class


def get_backend_class(backend_type: str) -> type[Backend]:
    """Get a backend class by type name.

    Args:
        backend_type: The type name of the backend.

    Returns:
        The backend class.

    Raises:
        ValueError: If the backend type is not registered.
    """
    if backend_type not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend type: {backend_type}. Available: {available}"
        )
    return _BACKEND_REGISTRY[backend_type]


def create_backend(config: dict[str, Any]) -> Backend:
    """Create a backend from configuration.

    Args:
        config: Configuration dict with at least a 'type' key.

    Returns:
        Configured backend instance.

    Raises:
        ValueError: If 'type' is missing or unknown.

    Example:
        backend = create_backend({
            "type": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "model": "qwen2.5-72b",
        })
    """
    backend_type = config.get("type")
    if not backend_type:
        raise ValueError("Backend config must include 'type' key")

    # Lazy registration of anthropic backend
    if backend_type == "anthropic" and "anthropic" not in _BACKEND_REGISTRY:
        try:
            from llm_infer.client.backends.anthropic import AnthropicBackend

            register_backend("anthropic", AnthropicBackend)
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install llm-infer[anthropic]"
            ) from e

    backend_class = get_backend_class(backend_type)
    return backend_class.from_config(config)


__all__ = [
    "Backend",
    "OpenAICompatibleBackend",
    "create_backend",
    "get_backend_class",
    "register_backend",
]
