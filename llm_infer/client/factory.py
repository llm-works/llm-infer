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

from appinfra.log import Logger

from .backends.base import Backend
from .backends.openai import OpenAICompatibleBackend
from .client import LLMClient
from .discovery import ModelDiscovery
from .router import LLMRouter

if TYPE_CHECKING:
    from appinfra.rate_limit import Backoff, RateLimiter

    from .strategy import RoutingStrategy


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
                config, discover_models, rate_limit_config, retry_config, strategy
            )

        return self._create_multi_backend_router(
            backends_config,
            config.get("default"),
            discover_models,
            rate_limit_config,
            retry_config,
            strategy,
        )

    def _create_single_backend_router(
        self,
        config: dict[str, Any],
        discover_models: bool,
        rate_limit_config: dict[str, Any] | None = None,
        retry_config: dict[str, Any] | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> LLMRouter:
        """Create router wrapping a single backend config.

        On failure during router init, closes client before re-raising.
        """
        client = self._create_client(config, rate_limit_config, retry_config)
        try:
            name = "default"
            clients = {name: client}
            configs = {name: config}

            # Always load static mappings, but only enable lazy probing if requested
            md = ModelDiscovery(self._lg, clients, configs)
            return LLMRouter(
                self._lg,
                clients,
                name,
                model_to_backend=md.models,
                discovery=md if discover_models else None,
                strategy=strategy,
            )
        except Exception:
            client.close()
            raise

    def _create_multi_backend_router(
        self,
        backends_config: dict[str, dict[str, Any]],
        default_name: str | None,
        discover_models: bool,
        rate_limit_config: dict[str, Any] | None = None,
        retry_config: dict[str, Any] | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> LLMRouter:
        """Create router from multi-backend config.

        On failure during router init, closes all clients before re-raising.
        """
        clients, backend_configs = self._create_enabled_clients(
            backends_config, rate_limit_config, retry_config
        )
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
            # Always load static mappings, but only enable lazy probing if requested
            md = ModelDiscovery(self._lg, clients, backend_configs)
            return LLMRouter(
                self._lg,
                clients,
                default_name,
                model_to_backend=md.models,
                discovery=md if discover_models else None,
                strategy=strategy,
            )
        except Exception:
            self._close_clients_safely(clients)
            raise

    def _create_enabled_clients(
        self,
        backends_config: dict[str, dict[str, Any]],
        rate_limit_config: dict[str, Any] | None = None,
        retry_config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, LLMClient], dict[str, dict[str, Any]]]:
        """Create clients for all enabled backends.

        On failure, closes any already-created clients before re-raising.
        """
        clients: dict[str, LLMClient] = {}
        configs: dict[str, dict[str, Any]] = {}
        try:
            for name, config in backends_config.items():
                if config.get("enabled", True):
                    clients[name] = self._create_client(
                        config, rate_limit_config, retry_config
                    )
                    configs[name] = config
        except Exception:
            self._close_clients_safely(clients)
            raise
        return clients, configs

    def _close_clients_safely(self, clients: dict[str, LLMClient]) -> None:
        """Close all clients, handling individual failures gracefully."""
        for client in clients.values():
            try:
                client.close()
            except Exception as e:
                self._lg.warning(
                    "Error closing client during cleanup", extra={"exception": e}
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

    def _create_rate_limiter(
        self, rate_limit_config: dict[str, Any] | None
    ) -> RateLimiter:
        """Create rate limiter from config.

        Args:
            rate_limit_config: Rate limit configuration with per_minute.

        Returns:
            RateLimiter from config, or default (60/min) if not configured.
        """
        from appinfra.rate_limit import RateLimiter

        if rate_limit_config is None:
            self._lg.warning(
                "no rate_limit configured, using default 60/min",
                extra={"per_minute": 60},
            )
            return RateLimiter(self._lg, per_minute=60)

        per_minute = rate_limit_config.get("per_minute")
        if per_minute is None:
            self._lg.warning(
                "no rate_limit configured, using default 60/min",
                extra={"per_minute": 60},
            )
            return RateLimiter(self._lg, per_minute=60)

        return RateLimiter(self._lg, per_minute=per_minute)

    def _create_retry(
        self, retry_config: dict[str, Any] | None
    ) -> tuple[Backoff | None, float]:
        """Create retry backoff and timeout from config.

        Args:
            retry_config: Retry configuration with enabled, timeout, and backoff.

        Returns:
            Tuple of (backoff, timeout). Backoff is None if retry disabled.
        """
        if retry_config is None:
            return None, 0
        if not retry_config.get("enabled", True):
            return None, 0

        from appinfra.rate_limit import Backoff

        backoff_config = retry_config.get("backoff", {})
        backoff = Backoff(
            self._lg,
            base=backoff_config.get("base", 1.0),
            max_delay=backoff_config.get("max", 60.0),
        )
        timeout = retry_config.get("timeout", 0)

        return backoff, timeout

    def _merge_config(
        self,
        global_config: dict[str, Any] | None,
        backend_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Merge global config with backend-specific overrides.

        Backend config takes precedence over global config.
        """
        if global_config is None and backend_config is None:
            return None
        if global_config is None:
            return backend_config
        if backend_config is None:
            return global_config
        return {**global_config, **backend_config}

    def _create_client(
        self,
        config: dict[str, Any],
        rate_limit_config: dict[str, Any] | None = None,
        retry_config: dict[str, Any] | None = None,
    ) -> LLMClient:
        """Create single LLMClient from backend configuration.

        Args:
            config: Backend configuration with 'type' key.
            rate_limit_config: Optional global rate limit configuration.
            retry_config: Optional global retry configuration.

        Returns:
            Configured LLMClient instance.
        """
        backend = self.create_backend(config)

        # Merge global config with per-backend overrides
        merged_rate_limit = self._merge_config(
            rate_limit_config, config.get("rate_limit")
        )
        merged_retry = self._merge_config(retry_config, config.get("retry"))

        rate_limiter = self._create_rate_limiter(merged_rate_limit)
        backoff, timeout = self._create_retry(merged_retry)

        return LLMClient(
            lg=self._lg,
            backend=backend,
            default_model=config.get("model"),
            rate_limiter=rate_limiter,
            backoff=backoff,
            timeout=timeout,
        )

    def from_backend_config(self, config: dict[str, Any]) -> LLMClient:
        """Create LLMClient from single backend configuration.

        Args:
            config: Backend configuration with 'type' key.

        Returns:
            Configured LLMClient instance.
        """
        backend = self.create_backend(config)
        rate_limiter = self._create_rate_limiter(config.get("rate_limit"))
        backoff, timeout = self._create_retry(config.get("retry"))
        return LLMClient(
            lg=self._lg,
            backend=backend,
            default_model=config.get("model"),
            rate_limiter=rate_limiter,
            backoff=backoff,
            timeout=timeout,
        )

    def openai(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str | None = None,
        timeout: float = 120.0,
        rate_limit: dict[str, Any] = {"per_minute": 60},
    ) -> LLMClient:
        """Create LLMClient for OpenAI-compatible API.

        Works with OpenAI, llm-infer, vLLM, Ollama, and other compatible servers.

        Args:
            base_url: API base URL.
            model: Default model name.
            api_key: Optional API key.
            timeout: Request timeout in seconds.
            rate_limit: Rate limit config. Defaults to 60 requests/minute.

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
        rate_limiter = self._create_rate_limiter(rate_limit)
        return LLMClient(
            lg=self._lg, backend=backend, default_model=model, rate_limiter=rate_limiter
        )

    def anthropic(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        rate_limit: dict[str, Any] = {"per_minute": 60},
    ) -> LLMClient:
        """Create LLMClient for Anthropic Claude API.

        Requires: pip install llm-infer[anthropic]

        Args:
            model: Claude model name.
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided).
            max_tokens: Default max tokens for responses.
            timeout: Request timeout in seconds.
            rate_limit: Rate limit config. Defaults to 60 requests/minute.

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
        rate_limiter = self._create_rate_limiter(rate_limit)
        return LLMClient(
            lg=self._lg, backend=backend, default_model=model, rate_limiter=rate_limiter
        )
