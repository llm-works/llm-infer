"""Factory for creating LLMClient instances.

This module provides the Factory class that handles:
- Backend registration and lookup
- Lazy loading of optional backends (e.g., Anthropic)
- Creation from configuration dicts
- Typed factory methods for specific backends

Example:
    from appinfra.log import Logger

    lg = Logger("my-app")
    factory = Factory(lg)

    # Direct factory methods
    client = factory.openai(base_url="http://localhost:8000/v1")
    client = factory.anthropic(model="claude-sonnet-4-20250514")

    # From configuration
    config = {
        "default": "local",
        "backends": {
            "local": {"type": "openai_compatible", "base_url": "http://..."},
            "cloud": {"type": "anthropic", "model": "claude-sonnet-4-20250514"},
        },
    }
    client = factory.from_config(config)
"""

from __future__ import annotations

import importlib
from typing import Any, cast

from appinfra.log import Logger

from llm_infer.client.backends.base import Backend
from llm_infer.client.backends.openai import OpenAICompatibleBackend
from llm_infer.client.client import LLMClient


class Factory:
    """Factory for creating LLMClient instances.

    Centralizes all client construction logic:
    - Registry of backend types
    - Lazy loading of optional dependencies
    - Config parsing and validation
    - Typed factory methods for convenience

    The registry is class-level state, shared across all instances.
    """

    # Backend type registry: name -> class (shared across instances)
    _registry: dict[str, type[Backend]] = {
        "openai_compatible": OpenAICompatibleBackend,
        "openai": OpenAICompatibleBackend,
    }

    # Lazy-loaded backends: name -> (module_path, class_name, install_hint)
    _lazy_backends: dict[str, tuple[str, str, str]] = {
        "anthropic": (
            "llm_infer.client.backends.anthropic",
            "AnthropicBackend",
            "pip install llm-infer[anthropic]",
        ),
    }

    def __init__(self, lg: Logger) -> None:
        """Initialize the factory.

        Args:
            lg: Logger instance for backend logging.
        """
        self._lg = lg

    @classmethod
    def register(cls, name: str, backend_class: type[Backend]) -> None:
        """Register a backend type.

        Args:
            name: Name to register the backend under.
            backend_class: Backend class to register.
        """
        cls._registry[name] = backend_class

    @classmethod
    def _ensure_registered(cls, backend_type: str) -> type[Backend]:
        """Ensure a backend type is registered, loading lazily if needed.

        Args:
            backend_type: The backend type name.

        Returns:
            The backend class.

        Raises:
            ValueError: If backend type is unknown.
            ImportError: If lazy backend's package is not installed.
        """
        if backend_type in cls._registry:
            return cls._registry[backend_type]

        if backend_type in cls._lazy_backends:
            module_path, class_name, install_hint = cls._lazy_backends[backend_type]
            try:
                module = importlib.import_module(module_path)
                backend_class = cast(type[Backend], getattr(module, class_name))
                cls.register(backend_type, backend_class)
                return backend_class
            except ImportError as e:
                raise ImportError(
                    f"{backend_type} backend requires additional dependencies. "
                    f"Install with: {install_hint}"
                ) from e

        available = ", ".join(sorted(cls._registry.keys() | cls._lazy_backends.keys()))
        raise ValueError(
            f"Unknown backend type: {backend_type}. Available: {available}"
        )

    def create_backend(self, config: dict[str, Any]) -> Backend:
        """Create a backend instance from configuration.

        Args:
            config: Configuration dict with at least a 'type' key.

        Returns:
            Configured backend instance.

        Raises:
            ValueError: If 'type' key is missing or unknown.
        """
        backend_type = config.get("type")
        if not backend_type:
            raise ValueError("Backend config must include 'type' key")

        backend_class = self._ensure_registered(backend_type)
        return backend_class.from_config(self._lg, config)

    def from_config(self, config: dict[str, Any]) -> LLMClient:
        """Create LLMClient from configuration dict.

        Supports two formats:

        Multi-backend (with default selection):
            default: backend_name
            backends:
              backend_name:
                type: openai_compatible
                base_url: http://localhost:8000/v1
              another:
                type: anthropic

        Single-backend (no wrapper):
            type: openai_compatible
            base_url: http://localhost:8000/v1

        Args:
            config: Configuration dictionary.

        Returns:
            Configured LLMClient instance.

        Raises:
            ValueError: If configuration is invalid.
        """
        backends_config = config.get("backends", {})
        default_name = config.get("default")

        if not backends_config:
            # Single backend config (no "backends" wrapper)
            return self.from_backend_config(config)

        if not default_name:
            # Use first backend as default
            default_name = next(iter(backends_config.keys()))

        if default_name not in backends_config:
            raise ValueError(f"Default backend '{default_name}' not found in backends")

        backend_config = backends_config[default_name]
        backend = self.create_backend(backend_config)

        return LLMClient(backend=backend, default_model=backend_config.get("model"))

    def from_backend_config(self, config: dict[str, Any]) -> LLMClient:
        """Create LLMClient from single backend configuration.

        Args:
            config: Backend configuration with 'type' key.

        Returns:
            Configured LLMClient instance.
        """
        backend = self.create_backend(config)
        return LLMClient(backend=backend, default_model=config.get("model"))

    def openai(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> LLMClient:
        """Create LLMClient for OpenAI-compatible API.

        Works with OpenAI, llm-infer, vLLM, Ollama, and other compatible servers.

        Args:
            base_url: API base URL.
            model: Default model name.
            api_key: Optional API key.
            timeout: Request timeout in seconds.

        Returns:
            LLMClient configured for OpenAI-compatible API.
        """
        backend = OpenAICompatibleBackend(
            lg=self._lg,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        return LLMClient(backend=backend, default_model=model)

    def anthropic(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> LLMClient:
        """Create LLMClient for Anthropic Claude API.

        Requires: pip install llm-infer[anthropic]

        Args:
            model: Claude model name.
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided).
            max_tokens: Default max tokens for responses.
            timeout: Request timeout in seconds.

        Returns:
            LLMClient configured for Anthropic API.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        self._ensure_registered("anthropic")
        config = {
            "type": "anthropic",
            "model": model,
            "api_key": api_key,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        backend = self.create_backend(config)
        return LLMClient(backend=backend, default_model=model)
