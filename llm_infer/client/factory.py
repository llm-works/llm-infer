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

from .backends import Backend, BackendFactory, RetryConfig
from .client import LLMClient
from .discovery import ModelDiscovery
from .embedding import EmbeddingClient
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
        self,
        config: dict[str, Any],
        name: str = "default",
        callbacks: LLMCallbacks | None = None,
    ) -> LLMClient:
        """Create LLMClient from single backend configuration.

        Args:
            config: Backend configuration with 'type' key.
            name: Backend name (for discovery/routing).
            callbacks: Optional callbacks for request/response/error lifecycle events.

        Returns:
            Configured LLMClient instance.
        """
        backend = self.create_backend(name, config)
        return LLMClient(self._lg, backend, callbacks=callbacks)

    def openai(
        self,
        base_url: str = "http://localhost:8000/v1",
        default_model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        rate_limit: dict[str, Any] | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> LLMClient:
        """Create LLMClient for OpenAI-compatible API.

        Works with OpenAI, llm-infer, vLLM, Ollama, and other compatible servers.

        Args:
            base_url: API base URL.
            default_model: Default model name.
            api_key: Optional API key.
            timeout: Request timeout in seconds.
            rate_limit: Optional rate limit config (e.g., {"per_minute": 60}).
            callbacks: Optional callbacks for request/response/error lifecycle events.

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
        return self.from_backend_config(config, "openai", callbacks)

    def anthropic(
        self,
        default_model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        rate_limit: dict[str, Any] | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> LLMClient:
        """Create LLMClient for Anthropic Claude API.

        Requires: pip install llm-infer[anthropic]

        Args:
            default_model: Claude model name.
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided).
            max_tokens: Default max tokens for responses.
            timeout: Request timeout in seconds.
            rate_limit: Optional rate limit config (e.g., {"per_minute": 60}).
            callbacks: Optional callbacks for request/response/error lifecycle events.

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
        return self.from_backend_config(config, "anthropic", callbacks)

    def embeddings(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "default",
        api_key: str | None = None,
        timeout: float = 120.0,
        retry: RetryConfig | None = None,
    ) -> EmbeddingClient:
        """Create EmbeddingClient for OpenAI-compatible embeddings API.

        Works with OpenAI, llm-infer, vLLM, and other compatible servers.

        Args:
            base_url: API base URL for embeddings endpoint.
            model: Model name to send in requests.
            api_key: Optional API key for Authorization header.
            timeout: Request timeout in seconds.
            retry: Retry configuration for transient errors. None disables retry.

        Returns:
            EmbeddingClient configured for the embeddings API.
        """
        from .backends.embedding import OpenAIBackend

        backend = OpenAIBackend(
            self._lg,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        return EmbeddingClient(self._lg, backend, retry=retry)

    def embeddings_google(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
        timeout: float = 120.0,
        retry: RetryConfig | None = None,
    ) -> EmbeddingClient:
        """Create EmbeddingClient for Google Generative AI embeddings.

        Args:
            api_key: Google API key.
            model: Model name (default: gemini-embedding-001).
            task_type: Task type for optimized embeddings. One of:
                RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY,
                CLASSIFICATION, CLUSTERING, QUESTION_ANSWERING, FACT_VERIFICATION.
            timeout: Request timeout in seconds.
            retry: Retry configuration for transient errors. None disables retry.

        Returns:
            EmbeddingClient configured for Google embeddings.
        """
        from .backends.embedding import GoogleBackend
        from .backends.providers.google import GoogleEmbeddingTaskType

        backend = GoogleBackend(
            self._lg,
            api_key=api_key,
            model=model,
            task_type=GoogleEmbeddingTaskType(task_type),
            timeout=timeout,
        )
        return EmbeddingClient(self._lg, backend, retry=retry)

    def _parse_embedding_retry(self, cfg: DotDict) -> RetryConfig | None:
        """Parse retry config for embeddings."""
        if "retry" not in cfg:
            return None
        retry_cfg = cfg.retry
        return RetryConfig(
            base=retry_cfg.get("base", 1.0),
            max_delay=retry_cfg.get("max_delay", 60.0),
            factor=retry_cfg.get("factor", 2.0),
            timeout=retry_cfg.get("timeout", 120.0),
        )

    def embeddings_from_config(self, config: dict[str, Any]) -> EmbeddingClient:
        """Create EmbeddingClient from configuration dict.

        Config format:
            type: openai              # or "google"
            base_url: http://...      # Required for openai
            api_key: sk-...           # Optional for openai, required for google
            model: text-embedding-3-small
            timeout: 120.0            # Optional, default 120.0
            task_type: RETRIEVAL_DOCUMENT  # Google only
            retry:                    # Optional
              base: 1.0
              max_delay: 60
              timeout: 120

        Args:
            config: Configuration dict.

        Returns:
            Configured EmbeddingClient.

        Raises:
            ValueError: If type is unknown or required fields are missing.
        """
        cfg = DotDict(config)
        backend_type = cfg.get("type", "openai")
        timeout = cfg.get("timeout", 120.0)
        retry = self._parse_embedding_retry(cfg)

        if backend_type in ("openai", "openai_compatible"):
            if not cfg.get("base_url"):
                raise ValueError("base_url required for openai embedding backend")
            return self.embeddings(
                base_url=cfg.base_url,
                model=cfg.get("model", "default"),
                api_key=cfg.get("api_key"),
                timeout=timeout,
                retry=retry,
            )
        elif backend_type == "google":
            if not cfg.get("api_key"):
                raise ValueError("api_key required for google embedding backend")
            return self.embeddings_google(
                api_key=cfg.api_key,
                model=cfg.get("model", "gemini-embedding-001"),
                task_type=cfg.get("task_type", "RETRIEVAL_DOCUMENT"),
                timeout=timeout,
                retry=retry,
            )
        else:
            raise ValueError(f"Unknown embedding backend type: {backend_type}")
