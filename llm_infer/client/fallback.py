# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Fallback client for cross-provider model resilience.

FallbackClient wraps an LLMRouter and automatically falls back to equivalent
models when the primary model fails with transient errors.

Fallbacks are defined as pairs (A -> B), which chain implicitly:
    gpt-4o -> claude-sonnet -> gemini-pro

Keys and values accept the ``model@backend`` syntax to pin a fallback step
to a specific backend when the same model is served by more than one:

    fallbacks = {
        "gpt-4o": "claude-sonnet-4-20250514@anthropic",
        "claude-sonnet-4-20250514@anthropic": "gemini-2.0-pro",
    }

Bare model refs (no ``@``) are accepted without cross-backend probing.
Cross-backend collisions in declared configs are already caught upstream
by ``ModelDiscovery`` as ``ModelConflictError``; a bare ref that no backend
declares resolves at request time via the router's default. Pin with
``model@backend`` when explicit routing is required.

Example:
    from llm_infer.client import Factory, FallbackClient

    router = Factory(lg).from_config(config)
    fallbacks = {
        "gpt-4o": "claude-sonnet-4-20250514",
        "claude-sonnet-4-20250514": "gemini-2.0-pro",
        "gemini-2.0-flash": "claude-haiku-4-5-20251001",
    }
    client = FallbackClient(lg, router, fallbacks)

    # If gpt-4o fails, tries claude-sonnet, then gemini-pro
    response = client.chat(messages, model="gpt-4o")
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Self

from appinfra.log import Logger

from .base import ChatClient
from .client import LLMClient
from .errors import BackendError, ConfigError
from .fallback_helper import detect_cycles, parse_fallback_key
from .log_utils import fmt_error
from .router import LLMRouter, ResolvedTarget
from .strategy import DefaultTransientDetector, TransientAction, TransientDetector
from .types import (
    ChatRequest,
    ChatResponse,
    ChatStream,
    ChatStreamSync,
    ResponseHolder,
    _ChatStream,
    _ChatStreamSync,
)


class FallbackClient(ChatClient):
    """Client that wraps LLMRouter with automatic model fallback.

    When a request fails with a transient error (5xx, timeout, unavailable,
    429 rate limit), FallbackClient consults the fallback pairs and retries
    with equivalent models until one succeeds or the chain is exhausted.
    For 429s the inner retry layer (RetryHelper) backs off against the same
    model first; fallback engages only once that budget is exhausted. A
    backend configured without retry falls back on its first transient
    error (a warning is logged at construction).

    Fallbacks are defined as pairs that chain implicitly:
        {"A": "B", "B": "C"} means A -> B -> C

    Attributes:
        router: The underlying LLMRouter for routing requests.
        fallbacks: Model fallback pairs.
    """

    def __init__(
        self,
        lg: Logger,
        router: LLMRouter,
        fallbacks: Mapping[str, str],
        detector: TransientDetector | None = None,
    ) -> None:
        """Initialize fallback client.

        Args:
            lg: Logger instance.
            router: LLMRouter to wrap.
            fallbacks: Model fallback pairs. Keys and values accept two forms:
                ``"model"`` (bare — backend resolved from router) or
                ``"model@backend"`` (pinned to a specific backend). Example:
                ``{"gpt-4o": "claude-sonnet@anthropic"}``. Chains are implicit:
                if the fallback is itself a key, it will be tried after failing.
                Cycles (A->B->A) retry round-robin until one succeeds.
            detector: Custom transient error detector. Uses DefaultTransientDetector
                if not provided.

        Raises:
            ConfigError: If a ``model@backend`` reference names an unknown
                backend.
            ValueError: If a ref is malformed (empty model or backend in
                ``model@backend`` syntax).
        """
        self._lg = lg
        self._router = router
        self._fallbacks = fallbacks
        self._detector = detector or DefaultTransientDetector()

        detect_cycles(fallbacks, lg)
        self._validate_no_ambiguity()
        self._warn_backends_without_retry()

    @property
    def router(self) -> LLMRouter:
        """The underlying router."""
        return self._router

    @property
    def fallbacks(self) -> Mapping[str, str]:
        """Model fallback pairs."""
        return self._fallbacks

    def _validate_no_ambiguity(self) -> None:
        """Validate ``model@backend`` refs and reject malformed refs.

        Only qualified refs are checked against the configured backends.
        Bare refs are accepted without probing: cross-backend collisions in
        declared configs are already surfaced upstream by ``ModelDiscovery``
        as ``ModelConflictError``, and a bare ref that no backend declares
        resolves at request time via the router. Models discovered at runtime
        (via ``list_models``) are routed first-wins. This keeps wire-up
        quiet — ``Backend.list_models`` is not called during fallback
        validation.

        Skipped when the router does not expose the ``clients`` surface
        (e.g., test doubles).
        """
        _bare, qualified = self._collect_refs()
        clients = getattr(self._router, "clients", None)
        if not isinstance(clients, Mapping):
            return
        self._validate_qualified_backends(qualified, clients)

    def _collect_refs(self) -> tuple[set[str], set[tuple[str, str]]]:
        """Split every key and value in the fallback map into bare vs qualified."""
        bare: set[str] = set()
        qualified: set[tuple[str, str]] = set()
        for key, value in self._fallbacks.items():
            for ref in (key, value):
                model, backend = parse_fallback_key(ref)
                if backend is None:
                    bare.add(model)
                else:
                    qualified.add((model, backend))
        return bare, qualified

    def _validate_qualified_backends(
        self, qualified: set[tuple[str, str]], clients: Mapping[str, Any]
    ) -> None:
        """Raise ConfigError if any ``model@backend`` names an unknown backend."""
        for model, backend in qualified:
            if backend not in clients:
                available = sorted(clients.keys())
                raise ConfigError(
                    f"Fallback ref {model + '@' + backend!r} names unknown "
                    f"backend {backend!r}; available: {available}"
                )

    def _next_key(self, current_key: str, current_backend: str | None) -> str | None:
        """Look up the next fallback for ``current_key``.

        Tries ``f"{model}@{current_backend}"`` first (when the current key is
        bare and the router resolved a backend for this attempt), then falls
        back to ``current_key`` as-is. Lets callers mix bare and qualified
        entries in the same map — a qualified entry wins when both forms
        exist, otherwise the bare entry is used.
        """
        if current_backend is not None and "@" not in current_key:
            qualified = f"{current_key}@{current_backend}"
            if qualified in self._fallbacks:
                return self._fallbacks[qualified]
        return self._fallbacks.get(current_key)

    def _resolve_target(
        self, current_key: str | None
    ) -> tuple[LLMClient, ResolvedTarget, str]:
        """Split ``current_key`` and resolve to (client, resolved, effective_key).

        When ``current_key`` contains ``@``, the backend part is passed as an
        explicit override to the router (highest routing priority).

        When ``current_key`` is None (caller omitted model), returns the
        router-resolved model as ``effective_key`` so fallback lookup uses
        the actual model, not the string ``"None"``.
        """
        model, backend_hint = (
            parse_fallback_key(current_key) if current_key is not None else (None, None)
        )
        resolved = self._router.resolve(model=model, backend=backend_hint)
        client = self._router.get_client(backend=resolved.backend)
        if current_key is not None:
            effective_key = current_key
        elif resolved.model is not None:
            effective_key = resolved.model
        else:
            # No model specified and router has no default — fallback can't engage.
            # Use empty string; lookup will find nothing and chain exhausts.
            effective_key = ""
        return client, resolved, effective_key

    def _warn_backends_without_retry(self) -> None:
        """Warn for backends without retry config.

        Without an inner retry budget, the first transient error (429, 5xx,
        timeout, unavailable) from such a backend escalates straight to the
        fallback model — no same-model backoff. Skipped when the router does
        not expose a clients mapping (e.g. test doubles).
        """
        clients = getattr(self._router, "clients", None)
        if not isinstance(clients, Mapping):
            return
        for name, client in clients.items():
            backend = getattr(client, "backend", None)
            ctx = getattr(backend, "ctx", None) if backend else None
            retry = getattr(ctx, "retry", object()) if ctx else object()
            if retry is None:
                self._lg.warning(
                    "backend has no retry config; "
                    "fallback engages on first transient error",
                    extra={"backend": name},
                )

    def _should_fallback(self, error: BackendError) -> bool:
        """Check if error should trigger fallback.

        Both RETRY_NEXT (5xx, timeout, unavailable) and RETRY_SAME (429 rate
        limit) trigger fallback. A 429 only reaches this layer after the
        inner RetryHelper has exhausted its same-model backoff budget, so
        escalating to the fallback model is the only remaining way to keep
        the request alive. Backends without a retry config fall back on their
        first transient error (a warning is logged at construction).
        """
        action = self._detector.classify(error)
        return action in (TransientAction.RETRY_NEXT, TransientAction.RETRY_SAME)

    def _log_fallback(
        self,
        failed: str,
        fallback: str,
        error: BackendError,
        attempt: int,
    ) -> None:
        """Log fallback attempt with full context."""
        from .errors import BackendRequestError

        status_code = None
        if isinstance(error, BackendRequestError):
            status_code = error.status_code

        self._lg.warning(
            "model request failed, trying fallback",
            extra={
                "failed_model": failed,
                "fallback_model": fallback,
                "error_type": type(error).__name__,
                "status_code": status_code,
                "error": fmt_error(error),
                "attempt": attempt,
            },
        )

    def _log_chain_exhausted(self, model: str, error: BackendError) -> None:
        """Log when all fallback models have failed."""
        self._lg.error(
            "all fallback models failed",
            extra={
                "original_model": model,
                "final_error": fmt_error(error),
            },
        )

    def _next_or_exhaust(
        self,
        current_key: str,
        current_backend: str | None,
        original_key: str | None,
        error: BackendError,
    ) -> str:
        """Return next fallback key or raise if the chain is exhausted."""
        next_key = self._next_key(current_key, current_backend)
        if next_key is None:
            self._log_chain_exhausted(str(original_key), error)
            raise error
        return next_key

    # =========================================================================
    # Sync API
    # =========================================================================

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send chat request with automatic fallback on transient errors."""
        request = ChatRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            context=context,
            extra=kwargs or None,
        )
        return self._chat_with_fallback(request)

    def _chat_with_fallback(self, request: ChatRequest) -> ChatResponse:
        """Execute chat with fallback, following pairs until success or no fallback."""
        current_key: str | None = request.model
        original_key: str | None = None
        attempt = 0

        while True:
            attempt += 1
            client, resolved, effective_key = self._resolve_target(current_key)
            if original_key is None:
                original_key = effective_key
            req = dataclasses.replace(request, model=resolved.model)
            try:
                return client._chat(req)
            except BackendError as e:
                if not self._should_fallback(e):
                    raise
                next_key = self._next_or_exhaust(
                    effective_key, resolved.backend, original_key, e
                )
                self._log_fallback(effective_key, next_key, e, attempt)
                current_key = next_key

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatStreamSync:
        """Stream chat with fallback (only before streaming starts)."""
        request = ChatRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            context=context,
            extra=kwargs or None,
        )
        holder = ResponseHolder()
        return _ChatStreamSync(self._stream_with_fallback(request, holder), holder)

    def _stream_with_fallback(
        self, request: ChatRequest, holder: ResponseHolder
    ) -> Iterator[str]:
        """Execute streaming chat with fallback, following pairs."""
        current_key: str | None = request.model
        original_key: str | None = None
        attempt = 0

        while True:
            attempt += 1
            streamed = False
            client, resolved, effective_key = self._resolve_target(current_key)
            if original_key is None:
                original_key = effective_key
            req = dataclasses.replace(request, model=resolved.model)
            try:
                for token in client._chat_stream(req, holder):
                    streamed = True
                    yield token
                return
            except BackendError as e:
                if streamed or not self._should_fallback(e):
                    raise
                next_key = self._next_or_exhaust(
                    effective_key, resolved.backend, original_key, e
                )
                self._log_fallback(effective_key, next_key, e, attempt)
                current_key = next_key

    # =========================================================================
    # Async API
    # =========================================================================

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send async chat request with automatic fallback."""
        request = ChatRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            context=context,
            extra=kwargs or None,
        )
        return await self._chat_async_with_fallback(request)

    async def _chat_async_with_fallback(self, request: ChatRequest) -> ChatResponse:
        """Execute async chat with fallback, following pairs until success or no fallback."""
        current_key: str | None = request.model
        original_key: str | None = None
        attempt = 0

        while True:
            attempt += 1
            client, resolved, effective_key = self._resolve_target(current_key)
            if original_key is None:
                original_key = effective_key
            req = dataclasses.replace(request, model=resolved.model)
            try:
                return await client._chat_async(req)
            except BackendError as e:
                if not self._should_fallback(e):
                    raise
                next_key = self._next_or_exhaust(
                    effective_key, resolved.backend, original_key, e
                )
                self._log_fallback(effective_key, next_key, e, attempt)
                current_key = next_key

    def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatStream:
        """Stream async chat with fallback (only before streaming starts)."""
        request = ChatRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            context=context,
            extra=kwargs or None,
        )
        holder = ResponseHolder()
        return _ChatStream(self._stream_async_with_fallback(request, holder), holder)

    async def _stream_async_with_fallback(
        self, request: ChatRequest, holder: ResponseHolder
    ) -> AsyncIterator[str]:
        """Execute async streaming chat with fallback, following pairs."""
        current_key: str | None = request.model
        original_key: str | None = None
        attempt = 0

        while True:
            attempt += 1
            streamed = False
            client, resolved, effective_key = self._resolve_target(current_key)
            if original_key is None:
                original_key = effective_key
            req = dataclasses.replace(request, model=resolved.model)
            try:
                async for token in client._chat_stream_async(req, holder):
                    streamed = True
                    yield token
                return
            except BackendError as e:
                if streamed or not self._should_fallback(e):
                    raise
                next_key = self._next_or_exhaust(
                    effective_key, resolved.backend, original_key, e
                )
                self._log_fallback(effective_key, next_key, e, attempt)
                current_key = next_key

    # =========================================================================
    # Rate limiting
    # =========================================================================

    def can_call(self) -> bool:
        """Check if a call is allowed (delegates to router)."""
        return self._router.can_call()

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        """Close sync resources (delegates to router)."""
        self._router.close()

    async def aclose(self) -> None:
        """Close async resources (delegates to router)."""
        await self._router.aclose()

    def __enter__(self) -> Self:
        """Enter sync context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context manager."""
        self.close()

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self.aclose()
