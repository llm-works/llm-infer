"""Factory for creating LLMClient and LLMRouter instances.

This module provides the Factory class that handles:
- Backend registration and lookup
- Lazy loading of optional backends (e.g., Anthropic)
- Creation from configuration dicts
- Typed factory methods for specific backends

Example:
    from appinfra.log import Logger

    lg = Logger("my-app")
    factory = Factory(lg)

    # Direct factory methods - return LLMClient (single backend)
    client = factory.openai(base_url="http://localhost:8000/v1")
    client = factory.anthropic(model="claude-sonnet-4-20250514")

    # From configuration - returns LLMRouter (multi-backend support)
    config = {
        "default": "local",
        "backends": {
            "local": {"type": "openai_compatible", "base_url": "http://..."},
            "cloud": {"type": "anthropic", "model": "claude-sonnet-4-20250514"},
        },
    }
    router = factory.from_config(config)
    router.chat(messages)                    # Uses default backend
    router.chat(messages, backend="cloud")   # Routes to specific backend
"""

from __future__ import annotations

import importlib
from typing import Any, cast

from appinfra.log import Logger

from llm_infer.client.backends.base import Backend
from llm_infer.client.backends.openai import OpenAICompatibleBackend
from llm_infer.client.client import LLMClient
from llm_infer.client.router import LLMRouter


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

    def from_config(
        self, config: dict[str, Any], discover_models: bool = True
    ) -> LLMRouter:
        """Create LLMRouter from configuration dict.

        Supports two formats:

        Multi-backend (with default selection and routing):
            default: backend_name
            backends:
              backend_name:
                enabled: true        # Optional, defaults to true
                type: openai_compatible
                base_url: http://localhost:8000/v1
                models:              # Optional: restrict to these models
                  - llama-3.1-8b
              another:
                enabled: false       # Disabled backends are skipped
                type: anthropic

        Single-backend (wrapped in router for consistent API):
            type: openai_compatible
            base_url: http://localhost:8000/v1

        Model-based routing:
            When discover_models=True (default), the factory queries each backend
            for available models and builds a routing table. If config specifies
            a `models` list, only those models are allowed (validated against
            discovered models). If the same model appears in multiple backends,
            raises ValueError.

        Args:
            config: Configuration dictionary.
            discover_models: If True, query backends for available models
                and enable model-based routing. Defaults to True.

        Returns:
            Configured LLMRouter instance with all enabled backends.

        Raises:
            ValueError: If configuration is invalid, no backends are enabled,
                config models not found in backend, or model conflict detected.
        """
        backends_config = config.get("backends", {})

        if not backends_config:
            return self._create_single_backend_router(config, discover_models)

        return self._create_multi_backend_router(
            backends_config, config.get("default"), discover_models
        )

    def _create_single_backend_router(
        self, config: dict[str, Any], discover_models: bool
    ) -> LLMRouter:
        """Create router wrapping a single backend config.

        On failure during model discovery or router init, closes client before re-raising.
        """
        client = self._create_client(config)
        try:
            name = "default"
            model_to_backend = self._discover_models_for_backend(
                name, client, config, discover_models
            )
            return LLMRouter(self._lg, {name: client}, name, model_to_backend)
        except Exception:
            client.close()
            raise

    def _create_multi_backend_router(
        self,
        backends_config: dict[str, dict[str, Any]],
        default_name: str | None,
        discover_models: bool,
    ) -> LLMRouter:
        """Create router from multi-backend config.

        On failure during model routing, closes all clients before re-raising.
        """
        clients, backend_configs = self._create_enabled_clients(backends_config)
        if not clients:
            raise ValueError("No enabled backends in config")

        if default_name and default_name not in clients:
            self._close_clients_safely(clients)
            raise ValueError(
                f"Default backend '{default_name}' not found in enabled backends"
            )
        if not default_name:
            default_name = next(iter(clients.keys()))

        try:
            model_to_backend = self._build_model_routing(
                clients, backend_configs, discover_models
            )
            return LLMRouter(self._lg, clients, default_name, model_to_backend)
        except Exception:
            self._close_clients_safely(clients)
            raise

    def _create_enabled_clients(
        self, backends_config: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, LLMClient], dict[str, dict[str, Any]]]:
        """Create clients for all enabled backends.

        On failure, closes any already-created clients before re-raising.
        """
        clients: dict[str, LLMClient] = {}
        configs: dict[str, dict[str, Any]] = {}
        try:
            for name, config in backends_config.items():
                if config.get("enabled", True):
                    clients[name] = self._create_client(config)
                    configs[name] = config
        except Exception:
            self._close_clients_safely(clients)
            raise
        return clients, configs

    def _build_model_routing(
        self,
        clients: dict[str, LLMClient],
        backend_configs: dict[str, dict[str, Any]],
        discover_models: bool,
    ) -> dict[str, str]:
        """Build model-to-backend routing table.

        Args:
            clients: Backend name to client mapping.
            backend_configs: Backend name to config mapping.
            discover_models: Whether to query backends for models.

        Returns:
            Mapping from model ID to backend name.

        Raises:
            ValueError: If config model not in backend, or model conflict.
        """
        model_to_backend: dict[str, str] = {}

        for name, client in clients.items():
            config = backend_configs[name]
            models = self._discover_models_for_backend(
                name, client, config, discover_models
            )

            for model, backend_name in models.items():
                if model in model_to_backend:
                    conflict = model_to_backend[model]
                    raise ValueError(
                        f"Model '{model}' found in multiple backends: "
                        f"'{conflict}' and '{backend_name}'"
                    )
                model_to_backend[model] = backend_name

        return model_to_backend

    def _discover_models_for_backend(
        self,
        name: str,
        client: LLMClient,
        config: dict[str, Any],
        discover_models: bool,
    ) -> dict[str, str]:
        """Discover and validate models for a single backend."""
        config_models: list[str] | None = config.get("models")

        if not discover_models:
            if config_models:
                return {model: name for model in config_models}
            return {}

        discovered = self._query_backend_models(name, client)
        if config_models:
            self._validate_config_models(name, config_models, discovered)
            return {model: name for model in config_models}

        return {model: name for model in discovered}

    def _query_backend_models(self, name: str, client: LLMClient) -> set[str]:
        """Query backend for available models."""
        try:
            return set(client.backend.list_models())
        except Exception as e:
            self._lg.warning(
                f"Failed to discover models from backend '{name}'",
                extra={"exception": e},
            )
            return set()

    def _validate_config_models(
        self, name: str, config_models: list[str], discovered: set[str]
    ) -> None:
        """Validate config models exist in discovered set."""
        if not discovered:
            # Can't validate if discovery failed - warn and trust config
            self._lg.warning(
                f"Backend '{name}' model discovery failed; "
                f"trusting config models without validation: {config_models}"
            )
            return
        missing = set(config_models) - discovered
        if missing:
            raise ValueError(
                f"Backend '{name}' config specifies models not available: "
                f"{sorted(missing)}. Available: {sorted(discovered)}"
            )

    def _close_clients_safely(self, clients: dict[str, LLMClient]) -> None:
        """Close all clients, handling individual failures gracefully."""
        for client in clients.values():
            try:
                client.close()
            except Exception as e:
                self._lg.warning(
                    "Error closing client during cleanup", extra={"exception": e}
                )

    def _create_client(self, config: dict[str, Any]) -> LLMClient:
        """Create single LLMClient from backend configuration.

        Args:
            config: Backend configuration with 'type' key.

        Returns:
            Configured LLMClient instance.
        """
        backend = self.create_backend(config)
        return LLMClient(backend=backend, default_model=config.get("model"))

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
