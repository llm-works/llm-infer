"""Fallback client for cross-provider model resilience.

FallbackClient wraps an LLMRouter and automatically falls back to equivalent
models when the primary model fails with transient errors.

Fallbacks are defined as pairs (A -> B), which chain implicitly:
    gpt-4o -> claude-sonnet -> gemini-pro

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
from .errors import BackendError
from .fallback_helper import detect_cycles
from .log_utils import fmt_error
from .router import LLMRouter
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
            fallbacks: Model fallback pairs. Maps model name to its fallback
                model, e.g.: {"gpt-4o": "claude-sonnet-4-20250514"}
                Chains are implicit: if claude-sonnet also has a fallback,
                it will be tried after claude-sonnet fails.
                Cycles (A->B->A) retry round-robin until one succeeds.
            detector: Custom transient error detector. Uses DefaultTransientDetector
                if not provided.
        """
        self._lg = lg
        self._router = router
        self._fallbacks = fallbacks
        self._detector = detector or DefaultTransientDetector()

        detect_cycles(fallbacks, lg)
        self._warn_backends_without_retry()

    @property
    def router(self) -> LLMRouter:
        """The underlying router."""
        return self._router

    @property
    def fallbacks(self) -> Mapping[str, str]:
        """Model fallback pairs."""
        return self._fallbacks

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
            if client.backend.ctx.retry is None:
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
        the request alive.
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

    def _call_with_model(self, request: ChatRequest, model: str | None) -> ChatResponse:
        """Call router's internal client with specific model."""
        resolved = self._router.resolve(model=model)
        req = dataclasses.replace(request, model=resolved.model)
        return self._router.get_client(backend=resolved.backend)._chat(req)

    async def _call_with_model_async(
        self, request: ChatRequest, model: str | None
    ) -> ChatResponse:
        """Call router's internal client with specific model (async)."""
        resolved = self._router.resolve(model=model)
        req = dataclasses.replace(request, model=resolved.model)
        return await self._router.get_client(backend=resolved.backend)._chat_async(req)

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
        model = request.model
        original_model = model
        attempt = 0

        while True:
            attempt += 1
            try:
                return self._call_with_model(request, model)
            except BackendError as e:
                if not self._should_fallback(e):
                    raise

                # Look up fallback from pairs
                next_model = self._fallbacks.get(model) if model else None
                if next_model is None:
                    self._log_chain_exhausted(str(original_model), e)
                    raise

                self._log_fallback(str(model), next_model, e, attempt)
                model = next_model

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
        model = request.model
        original_model = model
        attempt = 0

        while True:
            attempt += 1
            streamed = False
            try:
                resolved = self._router.resolve(model=model)
                req = dataclasses.replace(request, model=resolved.model)
                client = self._router.get_client(backend=resolved.backend)
                for token in client._chat_stream(req, holder):
                    streamed = True
                    yield token
                return
            except BackendError as e:
                if streamed or not self._should_fallback(e):
                    raise

                # Look up fallback from pairs
                next_model = self._fallbacks.get(model) if model else None
                if next_model is None:
                    self._log_chain_exhausted(str(original_model), e)
                    raise

                self._log_fallback(str(model), next_model, e, attempt)
                model = next_model

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
        model = request.model
        original_model = model
        attempt = 0

        while True:
            attempt += 1
            try:
                return await self._call_with_model_async(request, model)
            except BackendError as e:
                if not self._should_fallback(e):
                    raise

                # Look up fallback from pairs
                next_model = self._fallbacks.get(model) if model else None
                if next_model is None:
                    self._log_chain_exhausted(str(original_model), e)
                    raise

                self._log_fallback(str(model), next_model, e, attempt)
                model = next_model

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
        model = request.model
        original_model = model
        attempt = 0

        while True:
            attempt += 1
            streamed = False
            try:
                resolved = self._router.resolve(model=model)
                req = dataclasses.replace(request, model=resolved.model)
                client = self._router.get_client(backend=resolved.backend)
                async for token in client._chat_stream_async(req, holder):
                    streamed = True
                    yield token
                return
            except BackendError as e:
                if streamed or not self._should_fallback(e):
                    raise

                # Look up fallback from pairs
                next_model = self._fallbacks.get(model) if model else None
                if next_model is None:
                    self._log_chain_exhausted(str(original_model), e)
                    raise

                self._log_fallback(str(model), next_model, e, attempt)
                model = next_model

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
