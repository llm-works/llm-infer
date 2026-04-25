"""Backend factory - creates backends from configuration."""

from __future__ import annotations

from typing import Any

from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from appinfra.rate_limit import RateLimiter

from .base import Backend, BackendContext, RetryConfig


class BackendFactory:
    """Creates backends from configuration."""

    def __init__(self, lg: Logger) -> None:
        self._lg = lg

    def create(self, name: str, config: DotDict) -> Backend:
        """Create a backend from configuration.

        Args:
            name: Backend name (for discovery/routing).
            config: Backend configuration with 'type' and backend-specific settings.

        Returns:
            Configured backend instance.

        Raises:
            ValueError: If backend type is unknown.
        """
        ctx = self._create_context(config)
        backend_type = config.get("type", "openai_compatible")
        default_model = config.get("default_model")

        if backend_type in ("openai_compatible", "openai"):
            return self._create_openai(name, ctx, default_model, config)
        elif backend_type == "anthropic":
            return self._create_anthropic(name, ctx, default_model, config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def _create_context(self, config: DotDict) -> BackendContext:
        """Create BackendContext from config."""
        return BackendContext(
            rate_limiter=self._create_rate_limiter(config),
            retry=self._create_retry_config(config),
            request_timeout=config.get("timeout", 120.0),
        )

    def _create_rate_limiter(self, config: DotDict) -> RateLimiter | None:
        """Create RateLimiter from config."""
        rate_cfg = config.get("rate_limit")
        if not rate_cfg:
            return None
        return RateLimiter(
            self._lg,
            per_minute=rate_cfg.get("per_minute", 60),
        )

    def _create_retry_config(self, config: DotDict) -> RetryConfig | None:
        """Create RetryConfig from config."""
        retry_cfg = config.get("retry")
        if not retry_cfg:
            return None
        return RetryConfig(
            base=retry_cfg.get("base", 1.0),
            factor=retry_cfg.get("factor", 2.0),
            max_delay=retry_cfg.get("max_delay", 60.0),
            timeout=retry_cfg.get("timeout", 0),
        )

    def _create_openai(
        self,
        name: str,
        ctx: BackendContext,
        default_model: str | None,
        config: DotDict,
    ) -> Backend:
        """Create OpenAI-compatible backend."""
        from .openai import OpenAICompatibleBackend

        return OpenAICompatibleBackend(
            lg=self._lg,
            name=name,
            ctx=ctx,
            default_model=default_model,
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            api_key=config.get("api_key"),
        )

    def _create_anthropic(
        self,
        name: str,
        ctx: BackendContext,
        default_model: str | None,
        config: DotDict,
    ) -> Backend:
        """Create Anthropic backend."""
        from .anthropic import AnthropicBackend

        kwargs: dict[str, Any] = {
            "lg": self._lg,
            "name": name,
            "ctx": ctx,
            "default_model": default_model,
            "api_key": config.get("api_key"),
            "base_url": config.get("base_url"),
        }
        if config.get("max_tokens") is not None:
            kwargs["max_tokens"] = config["max_tokens"]
        return AnthropicBackend(**kwargs)
