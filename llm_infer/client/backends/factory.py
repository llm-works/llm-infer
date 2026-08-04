"""Backend factory - creates backends from configuration."""

from __future__ import annotations

from typing import Any

from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from appinfra.rate_limit import RateLimiter
from appinfra.yaml import SecretStr

from .auth import AuthProvider, auth_from_config
from .base import Backend
from .context import BackendContext, RetryConfig
from .provider import Provider, ProviderDetector


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
        default_model = config.get("model")

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

    def _resolve_provider(
        self,
        name: str,
        config: DotDict,
        base_url: str,
        api_key: str | None,
    ) -> Provider:
        """Return the provider for this backend.

        Precedence: explicit ``config.provider`` wins; otherwise fall back
        to URL/key auto-detection via :class:`ProviderDetector`. When both
        yield a known value and disagree, WARN and honor the explicit
        setting — the user's config is authoritative, but the mismatch is
        surfaced for debugging (typical cause: wrong ``base_url``).
        """
        detected = ProviderDetector.detect(base_url, api_key)
        explicit_raw = config.get("provider")
        if explicit_raw is None:
            return detected
        try:
            explicit = Provider(explicit_raw)
        except ValueError as e:
            valid = sorted(p.value for p in Provider)
            raise ValueError(
                f"Backend {name!r}: invalid provider {explicit_raw!r}; "
                f"expected one of {valid}"
            ) from e
        if detected is not Provider.UNKNOWN and explicit is not detected:
            self._lg.warning(
                "backend provider explicit vs auto-detect mismatch; using explicit",
                extra={
                    "backend": name,
                    "provider": {
                        "explicit": explicit.value,
                        "detected": detected.value,
                    },
                },
            )
        return explicit

    def _create_openai(
        self,
        name: str,
        ctx: BackendContext,
        default_model: str | None,
        config: DotDict,
    ) -> Backend:
        """Create OpenAI-compatible backend.

        Resolves the provider (explicit ``config.provider`` if set, else
        URL/key auto-detection) and returns a specialized backend when the
        provider has one (e.g. GeminiBackend for Google).
        """
        base_url = config.get("base_url", "http://localhost:8000/v1")
        api_key = SecretStr.ensure(config.get("api_key"))
        auth = self._create_auth(config, api_key=api_key)
        # Provider detection uses the raw prefix; reveal into a local only.
        detect_key = api_key.reveal() if api_key is not None else None
        provider = self._resolve_provider(name, config, base_url, detect_key)

        kwargs: dict[str, Any] = {
            "lg": self._lg,
            "name": name,
            "ctx": ctx,
            "default_model": default_model,
            "base_url": base_url,
            "auth": auth,
            "provider": provider,
        }

        if provider == Provider.GOOGLE:
            from .providers.gemini import GeminiBackend

            service_tier = config.get("service_tier")
            if service_tier is not None:
                kwargs["service_tier"] = service_tier
            return GeminiBackend(**kwargs)

        from .providers.openai import OpenAICompatibleBackend

        return OpenAICompatibleBackend(**kwargs)

    def _create_auth(
        self,
        config: DotDict,
        *,
        api_key: str | SecretStr | None,
        api_key_header: str = "Authorization",
    ) -> AuthProvider | None:
        """Build an AuthProvider from the ``auth:`` block, or wrap ``api_key``."""
        auth_cfg = config.get("auth")
        return auth_from_config(
            self._lg,
            dict(auth_cfg) if auth_cfg else None,
            api_key=api_key,
            api_key_header=api_key_header,
        )

    def _create_anthropic(
        self,
        name: str,
        ctx: BackendContext,
        default_model: str | None,
        config: DotDict,
    ) -> Backend:
        """Create Anthropic backend."""
        from .providers.anthropic import AnthropicBackend

        kwargs: dict[str, Any] = {
            "lg": self._lg,
            "name": name,
            "ctx": ctx,
            "default_model": default_model,
            "api_key": SecretStr.ensure(config.get("api_key")),
            "base_url": config.get("base_url"),
        }
        if config.get("max_tokens") is not None:
            kwargs["max_tokens"] = config["max_tokens"]
        if config.get("thinking_budget") is not None:
            kwargs["thinking_budget"] = config["thinking_budget"]
        return AnthropicBackend(**kwargs)
