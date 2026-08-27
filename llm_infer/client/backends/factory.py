# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Backend factory - creates backends from configuration."""

from __future__ import annotations

import re
from typing import Any

from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from appinfra.yaml import SecretStr

from ..types import LLMCallbacks
from .auth import AuthProvider, auth_from_config
from .base import Backend
from .context import BackendContext, context_from_config
from .provider import Provider, ProviderDetector
from .vertex_native import NativeVertexBackend, NativeVertexFactory

NON_CHAT_BACKEND_TYPES: frozenset[str] = frozenset({"vertex_native"})

_VERTEX_REGION_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$|^global$")
"""Valid Vertex AI region: lowercase alphanumeric with hyphens, or 'global'.

Rejects URL-delimiter characters that could enable SSRF via hostname
interpolation (e.g. 'attacker.example/#' would route requests off-GCP)."""
"""Backend ``type:`` values that are sibling to :class:`Backend` rather than
implementations of it — routed through dedicated ``Factory.<surface>s_from_config()``
accessors, not through :class:`~..router.LLMRouter`."""


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
            ValueError: If backend type is unknown, or when called with a non-chat
                ``type:`` (see :data:`NON_CHAT_BACKEND_TYPES`) that requires a
                dedicated accessor (e.g. :meth:`create_vertex_native`).
        """
        ctx = self._create_context(config)
        backend_type = config.get("type", "openai_compatible")
        default_model = config.get("model")

        if backend_type in ("openai_compatible", "openai"):
            return self._create_openai(name, ctx, default_model, config)
        elif backend_type == "anthropic":
            return self._create_anthropic(name, ctx, default_model, config)
        elif backend_type in NON_CHAT_BACKEND_TYPES:
            raise ValueError(
                f"Backend {name!r} has non-chat type {backend_type!r}; "
                f"use Factory.vertex_natives_from_config() to build these entries"
            )
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def create_vertex_native(
        self,
        name: str,
        config: DotDict,
        *,
        callbacks: LLMCallbacks | None = None,
    ) -> NativeVertexBackend:
        """Create a :class:`NativeVertexBackend` from a yaml block.

        Symmetric with :meth:`create` for chat backends, but returns the
        sibling (non-chat) native Vertex surface. ``project`` and ``region``
        are read from the yaml block — they were kwargs before PR #138 when
        the design still assumed one Vertex block could serve both the
        chat and native paths; ground truth was that consumers deep-copied
        the block with per-path overrides, so both paths now own their own
        block.

        Args:
            name: Backend name — used in error messages and log ``extra``.
            config: Backend yaml block. Must contain ``project``, ``region``,
                and an ``auth`` sub-block; may include ``service_tier``,
                ``rate_limit``, ``retry``, ``timeout``.
            callbacks: Optional lifecycle callbacks (retry / error).

        Raises:
            ValueError: If ``project`` or ``region`` is missing / empty,
                if ``region`` contains invalid characters (SSRF prevention),
                or if the ``auth`` sub-block is missing (native Vertex requires
                SA credentials).
        """
        project = config.get("project")
        region = config.get("region")
        if not project or not region:
            raise ValueError(
                f"Backend {name!r} (type: vertex_native): 'project' and 'region' "
                f"are required yaml fields "
                f"(got project={project!r}, region={region!r})"
            )
        if not _VERTEX_REGION_RE.match(region):
            raise ValueError(
                f"Backend {name!r}: invalid region {region!r}. "
                f"Must be 'global' or lowercase alphanumeric with hyphens "
                f"(e.g. 'us-central1')"
            )
        return NativeVertexFactory(self._lg).create(
            config, project=project, region=region, callbacks=callbacks
        )

    def _create_context(self, config: DotDict) -> BackendContext:
        """Create BackendContext from config."""
        return context_from_config(self._lg, config)

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
