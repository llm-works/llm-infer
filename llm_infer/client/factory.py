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
from typing import TYPE_CHECKING, Any, cast

from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from .backends import Backend, BackendFactory
from .client import LLMClient
from .discovery import ModelDiscovery
from .router import LLMRouter
from .types import LLMCallbacks

if TYPE_CHECKING:
    from .strategy import RoutingStrategy


class Factory:
    """Factory for creating LLMClient and LLMRouter instances.

    Centralizes all client construction logic:
    - Backend creation via BackendFactory
    - Config parsing and validation
    - Typed factory methods for convenience
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize the factory.

        Args:
            lg: Logger instance for backend logging.
        """
        self._lg = lg
        self._backend_factory = BackendFactory(lg)

    def create_backend(self, name: str, config: dict[str, Any]) -> Backend:
        """Create a backend instance from configuration.

        Args:
            name: Backend name (for discovery/routing).
            config: Configuration dict with backend-specific settings.

        Returns:
            Configured backend instance.
        """
        return self._backend_factory.create(name, DotDict(config))

    def from_config(
        self,
        config: dict[str, Any],
        discover_models: bool = True,
        callbacks: LLMCallbacks | None = None,
    ) -> LLMRouter:
        """Create LLMRouter from configuration dict.

        Supports two formats:

        Multi-backend (with default selection and routing):
            default: backend_name
            rate_limit:              # Optional: rate limiting for all backends
              per_minute: 60
            retry:                   # Optional: retry with exponential backoff
              backoff:
                base: 1.0
                max: 60
              timeout: 120           # Optional: request timeout in seconds
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
            When discover_models=True (default), the router performs LAZY model
            discovery - backends are only probed when first used. If config
            specifies a `models` list, those models are used directly without
            probing the backend. If the same model appears in multiple backends,
            raises ValueError.

        Rate limiting:
            When rate_limit is specified, each backend gets its own rate limiter.
            - per_minute: Maximum requests per minute per backend

        Retry:
            When retry is specified, failed requests use exponential backoff.
            - backoff.base: Initial backoff delay in seconds (default: 1.0)
            - backoff.max: Maximum backoff delay in seconds (default: 60.0)
            - timeout: Request timeout in seconds (default: 0, no timeout)

        Args:
            config: Configuration dictionary.
            discover_models: If True, enable lazy model discovery when backends
                are first used. Defaults to True.
            callbacks: Optional callbacks for request/response/error lifecycle
                events. Applied to all clients created by this router.

        Returns:
            Configured LLMRouter instance with all enabled backends.

        Raises:
            ValueError: If configuration is invalid or no backends are enabled.
            ModelConflictError: If the same model appears in multiple backend configs.
        """
        backends_config = config.get("backends", {})
        rate_limit_config = config.get("rate_limit")
        retry_config = config.get("retry")
        strategy = self._create_strategy(config.get("strategy"))

        if not backends_config:
            return self._create_single_backend_router(
                config,
                discover_models,
                rate_limit_config,
                retry_config,
                strategy,
                callbacks,
            )

        return self._create_multi_backend_router(
            backends_config,
            config.get("default"),
            discover_models,
            rate_limit_config,
            retry_config,
            strategy,
            callbacks,
        )

    def _create_single_backend_router(
        self,
        config: dict[str, Any],
        discover_models: bool,
        rate_limit_config: dict[str, Any] | None = None,
        retry_config: dict[str, Any] | None = None,
        strategy: RoutingStrategy | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> LLMRouter:
        """Create router wrapping a single backend config.

        On failure during router init, closes backend/client before re-raising.
        """
        name = "default"
        merged_config = self._merge_backend_config(
            config, rate_limit_config, retry_config
        )
        backend = self.create_backend(name, merged_config)
        backends = {name: backend}
        configs = {name: DotDict(merged_config)}

        try:
            discovery = ModelDiscovery(self._lg, backends, configs)
            client = LLMClient(self._lg, backend, discovery, callbacks)
            clients = {name: client}

            return LLMRouter(
                self._lg,
                clients,
                name,
                discovery=discovery,
                strategy=strategy,
            )
        except Exception:
            backend.close()
            raise

    def _create_multi_backend_router(  # cq: max-lines=35
        self,
        backends_config: dict[str, dict[str, Any]],
        default_name: str | None,
        discover_models: bool,
        rate_limit_config: dict[str, Any] | None = None,
        retry_config: dict[str, Any] | None = None,
        strategy: RoutingStrategy | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> LLMRouter:
        """Create router from multi-backend config.

        On failure during router init, closes all backends/clients before re-raising.
        """
        backends, configs = self._create_enabled_backends(
            backends_config, rate_limit_config, retry_config
        )
        if not backends:
            raise ValueError("No enabled backends in config")

        if default_name and default_name not in backends:
            self._close_backends_safely(backends)
            raise ValueError(
                f"Default backend '{default_name}' not found in enabled backends"
            )
        if not default_name:
            default_name = next(iter(backends.keys()))

        try:
            discovery = ModelDiscovery(self._lg, backends, configs)
            clients = {
                name: LLMClient(self._lg, backend, discovery, callbacks)
                for name, backend in backends.items()
            }

            return LLMRouter(
                self._lg,
                clients,
                default_name,
                discovery=discovery,
                strategy=strategy,
            )
        except Exception:
            self._close_backends_safely(backends)
            raise

    def _create_enabled_backends(
        self,
        backends_config: dict[str, dict[str, Any]],
        rate_limit_config: dict[str, Any] | None = None,
        retry_config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Backend], dict[str, DotDict]]:
        """Create backends for all enabled backend configs.

        On failure, closes any already-created backends before re-raising.
        """
        backends: dict[str, Backend] = {}
        configs: dict[str, DotDict] = {}
        try:
            for name, config in backends_config.items():
                if config.get("enabled", True):
                    merged = self._merge_backend_config(
                        config, rate_limit_config, retry_config
                    )
                    backends[name] = self.create_backend(name, merged)
                    configs[name] = DotDict(merged)
        except Exception:
            self._close_backends_safely(backends)
            raise
        return backends, configs

    def _merge_backend_config(
        self,
        config: dict[str, Any],
        rate_limit_config: dict[str, Any] | None,
        retry_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge global rate_limit/retry config with per-backend config."""
        result = dict(config)
        if rate_limit_config and "rate_limit" not in result:
            result["rate_limit"] = rate_limit_config
        elif rate_limit_config and "rate_limit" in result:
            result["rate_limit"] = {**rate_limit_config, **result["rate_limit"]}
        if retry_config and "retry" not in result:
            result["retry"] = retry_config
        elif retry_config and "retry" in result:
            result["retry"] = {**retry_config, **result["retry"]}
        return result

    def _close_backends_safely(self, backends: dict[str, Backend]) -> None:
        """Close all backends, handling individual failures gracefully."""
        for backend in backends.values():
            try:
                backend.close()
            except Exception as e:
                self._lg.warning(
                    "Error closing backend during cleanup", extra={"exception": e}
                )

    def _create_strategy(  # cq: max-lines=40
        self, strategy_config: dict[str, Any] | None
    ) -> RoutingStrategy | None:
        """Create routing strategy from config.

        Config formats:
            Built-in strategies:
                strategy:
                  type: fallback
                  order: [primary, fallback]
                  roles:
                    synthesis: [gpt4, claude]

            Custom factory (module must export Factory class):
                strategy:
                  factory: mypackage.module
                  custom_option: value

        Args:
            strategy_config: Strategy configuration dict.

        Returns:
            Configured strategy, or None if no config.
        """
        if not strategy_config:
            return None

        from appinfra.dot_dict import DotDict

        from .strategies import (
            DefaultStrategyFactory,
            FallbackStrategyFactory,
        )

        config = DotDict(strategy_config)

        # Custom factory from package (expects module.Factory class)
        if "factory" in config:
            from .strategy import StrategyFactory

            module = importlib.import_module(config.factory)
            factory = cast(StrategyFactory, module.Factory())
            return factory.create(self._lg, config)

        # Built-in strategies
        strategy_type = config.get("type", "default")
        factories: dict[
            str, type[DefaultStrategyFactory] | type[FallbackStrategyFactory]
        ] = {
            "default": DefaultStrategyFactory,
            "fallback": FallbackStrategyFactory,
        }

        if strategy_type not in factories:
            raise ValueError(
                f"Unknown strategy type '{strategy_type}'. "
                f"Available: {list(factories.keys())}"
            )

        return factories[strategy_type]().create(self._lg, config)

    def from_backend_config(
        self, config: dict[str, Any], name: str = "default"
    ) -> LLMClient:
        """Create LLMClient from single backend configuration.

        Args:
            config: Backend configuration with 'type' key.
            name: Backend name (for discovery/routing).

        Returns:
            Configured LLMClient instance.
        """
        backend = self.create_backend(name, config)
        return LLMClient(self._lg, backend)

    def openai(
        self,
        base_url: str = "http://localhost:8000/v1",
        default_model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        rate_limit: dict[str, Any] | None = None,
    ) -> LLMClient:
        """Create LLMClient for OpenAI-compatible API.

        Works with OpenAI, llm-infer, vLLM, Ollama, and other compatible servers.

        Args:
            base_url: API base URL.
            default_model: Default model name.
            api_key: Optional API key.
            timeout: Request timeout in seconds.
            rate_limit: Optional rate limit config (e.g., {"per_minute": 60}).

        Returns:
            LLMClient configured for OpenAI-compatible API.
        """
        config: dict[str, Any] = {
            "type": "openai_compatible",
            "base_url": base_url,
            "model": default_model,
            "api_key": api_key,
            "timeout": timeout,
        }
        if rate_limit:
            config["rate_limit"] = rate_limit
        return self.from_backend_config(config, "openai")

    def anthropic(
        self,
        default_model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        rate_limit: dict[str, Any] | None = None,
    ) -> LLMClient:
        """Create LLMClient for Anthropic Claude API.

        Requires: pip install llm-infer[anthropic]

        Args:
            default_model: Claude model name.
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided).
            max_tokens: Default max tokens for responses.
            timeout: Request timeout in seconds.
            rate_limit: Optional rate limit config (e.g., {"per_minute": 60}).

        Returns:
            LLMClient configured for Anthropic API.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        config: dict[str, Any] = {
            "type": "anthropic",
            "model": default_model,
            "api_key": api_key,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if rate_limit:
            config["rate_limit"] = rate_limit
        return self.from_backend_config(config, "anthropic")
