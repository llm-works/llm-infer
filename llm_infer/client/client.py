"""Single-backend LLM client.

This module provides the LLMClient class that wraps a single backend.
For multi-backend routing, use LLMRouter.

For client creation, use Factory:
    from appinfra.log import Logger
    from llm_infer.client import Factory

    lg = Logger("my-app")
    factory = Factory(lg)

    # Quick start with OpenAI-compatible server
    with factory.openai(base_url="http://localhost:8000/v1") as client:
        response = client.chat([{"role": "user", "content": "Hello"}])
        print(response)

    # Async streaming with Anthropic
    async with factory.anthropic() as client:
        async for token in client.chat_stream_async(messages):
            print(token, end="")

    # Multi-backend config returns LLMRouter
    router = factory.from_config(config)
    router.chat(messages, backend="openai")  # Route to specific backend
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from copy import copy
from typing import Any, Self

from appinfra.log import Logger
from appinfra.time import since, start

from ..schemas.openai import ChatCompletionUsage
from .backends import Backend
from .base import ChatClient
from .discovery import ModelDiscovery
from .retry import RetryHelper
from .types import (
    ChatRequest,
    ChatResponse,
    ChatStream,
    ChatStreamSync,
    LLMCallbacks,
    ResponseHolder,
    _ChatStream,
    _ChatStreamSync,
)


def _tokens_log(usage: ChatCompletionUsage | None) -> dict[str, int] | None:
    """Build the tokens sub-dict for response logs.

    Includes ``cached`` (prompt tokens served from the provider's prompt cache)
    so cache behaviour is visible by default. Defaults to 0 when the provider
    did not report prompt_tokens_details.
    """
    if usage is None:
        return None
    details = usage.prompt_tokens_details
    cached = details.cached_tokens if details else 0
    return {
        "in": usage.prompt_tokens,
        "out": usage.completion_tokens,
        "cached": cached,
    }


class LLMClient(ChatClient):
    """Single-backend LLM client with sync/async support.

    This client wraps a single backend and provides a consistent interface
    for chat completions. For multi-backend routing, use LLMRouter.

    The client delegates all operations to an underlying Backend instance,
    which handles the actual API communication. Rate limiting and retry
    configuration are owned by the backend (via BackendContext).

    Create instances using Factory:
        from appinfra.log import Logger
        from llm_infer.client import Factory

        lg = Logger("my-app")
        factory = Factory(lg)

        # Sync
        with factory.openai() as client:
            response = client.chat(messages)

        # Async
        async with factory.anthropic() as client:
            response = await client.chat_async(messages)

    Attributes:
        backend: The underlying backend instance.
    """

    def __init__(
        self,
        lg: Logger,
        backend: Backend,
        discovery: ModelDiscovery | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> None:
        """Initialize the client with a backend.

        Args:
            lg: Logger instance.
            backend: The backend to use for API calls. The backend owns
                rate limiting and retry configuration via BackendContext.
            discovery: Optional ModelDiscovery for resolving 'auto'/'default' model
                names. When provided, handles model resolution including backend probing.
                When None, falls back to backend's default_model.
            callbacks: Optional callbacks for request/response/error lifecycle events.
        """
        self._lg = lg
        self._backend = backend
        self._discovery = discovery
        self._retry = RetryHelper(lg, backend.ctx, backend.provider)
        self._callbacks = callbacks

    @property
    def backend(self) -> Backend:
        """The underlying backend instance."""
        return self._backend

    @property
    def default_model(self) -> str | None:
        """Default model used when not specified per-request."""
        return self._backend.default_model

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats.

        This is populated after both streaming and non-streaming requests,
        providing access to usage statistics and metadata.
        """
        return self._backend.last_response

    @property
    def discovery(self) -> ModelDiscovery | None:
        """Model discovery for 'auto'/'default' resolution, if configured."""
        return self._discovery

    def resolve_model(self, model: str | None) -> str | None:
        """Resolve model name using discovery or fallback to default_model.

        Args:
            model: Model name from request (may be None, 'auto', or 'default').

        Returns:
            Resolved model name.
        """
        if self._discovery is not None:
            return self._discovery.resolve_model(
                self._backend.name, model, self._backend.default_model
            )
        # Fallback: simple default_model substitution (no 'auto' support)
        return model if model is not None else self._backend.default_model

    def can_call(self) -> bool:
        """Check if a call would be allowed right now (non-blocking).

        Returns False if rate limited. Delegates to backend.can_call().

        Returns:
            True if a call would be allowed, False otherwise.
        """
        return self._backend.can_call()

    def with_callbacks(self, callbacks: LLMCallbacks) -> Self:
        """Return a client copy with callbacks configured.

        Callbacks fire on request/response/error events for cost tracking,
        logging, tracing, or metrics collection.

        Args:
            callbacks: Callbacks for lifecycle events.

        Returns:
            New client instance with callbacks configured.
        """
        clone = copy(self)
        clone._callbacks = callbacks
        return clone

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
        """Send a chat completion request (sync).

        When backoff is configured, automatically retries on transient errors
        (connection failures and HTTP 5xx/429/529) with exponential backoff.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            think: Enable thinking mode.
            adapter: LoRA adapter name (OpenAI-compatible only).
            context: User context passed to callbacks (cost tracking, tracing).
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """
        request = ChatRequest(
            messages=messages,
            model=self.resolve_model(model),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            extra=kwargs or None,
            context=context,
        )
        return self._chat(request)

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
        """Stream chat completion tokens (sync).

        Transient errors are retried with backoff until the first token is
        yielded (same policy as non-streaming); after that, errors propagate.

        Returns a ChatStreamSync that yields tokens. After iteration, access
        stream.response for usage statistics.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            think: Enable thinking mode.
            adapter: LoRA adapter name (OpenAI-compatible only).
            context: User context passed to callbacks (cost tracking, tracing).
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatStreamSync that yields tokens and provides response after completion.
        """
        request = ChatRequest(
            messages=messages,
            model=self.resolve_model(model),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            extra=kwargs or None,
            context=context,
        )
        holder = ResponseHolder()
        return _ChatStreamSync(self._chat_stream(request, holder), holder)

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
        """Send a chat completion request (async).

        When backoff is configured, automatically retries on transient errors
        (connection failures and HTTP 5xx/429/529) with exponential backoff.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            think: Enable thinking mode.
            adapter: LoRA adapter name (OpenAI-compatible only).
            context: User context passed to callbacks (cost tracking, tracing).
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """
        request = ChatRequest(
            messages=messages,
            model=self.resolve_model(model),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            extra=kwargs or None,
            context=context,
        )
        return await self._chat_async(request)

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
        """Stream chat completion tokens (async).

        Transient errors are retried with backoff until the first token is
        yielded (same policy as non-streaming); after that, errors propagate.

        Returns a ChatStream that yields tokens. After iteration, access
        stream.response for usage statistics.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            think: Enable thinking mode.
            adapter: LoRA adapter name (OpenAI-compatible only).
            context: User context passed to callbacks (cost tracking, tracing).
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatStream that yields tokens and provides response after completion.
        """
        request = ChatRequest(
            messages=messages,
            model=self.resolve_model(model),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            extra=kwargs or None,
            context=context,
        )
        holder = ResponseHolder()
        return _ChatStream(self._chat_stream_async(request, holder), holder)

    # =========================================================================
    # Internal API (for Router - takes ChatRequest directly)
    # =========================================================================

    def _get_stream_response(
        self, holder: ResponseHolder | None
    ) -> ChatResponse | None:
        """Get response from holder or fall back to backend.last_response."""
        if holder is not None:
            return holder.value
        self._lg.warning(
            "falling back to backend.last_response (not thread-safe for concurrent streams)",
            extra={"backend": self._backend.name, "provider": self._backend.provider},
        )
        return self._backend.last_response

    def _fire_on_request(self, request: ChatRequest) -> None:
        """Fire on_request callback with error handling."""
        cb = self._callbacks
        if cb and cb.on_request:
            try:
                cb.on_request(request, 0)
            except Exception as e:
                self._lg.warning("on_request callback failed", extra={"exception": e})

    def _fire_on_response(self, request: ChatRequest, response: ChatResponse) -> None:
        """Fire on_response callback with error handling."""
        cb = self._callbacks
        if cb and cb.on_response:
            try:
                cb.on_response(request, response)
            except Exception as e:
                self._lg.warning("on_response callback failed", extra={"exception": e})

    def _fire_on_error(self, request: ChatRequest, error: Exception) -> None:
        """Fire on_error callback with error handling."""
        cb = self._callbacks
        if cb and cb.on_error:
            try:
                cb.on_error(request, error)
            except Exception as cb_err:
                self._lg.warning(
                    "on_error callback failed", extra={"exception": cb_err}
                )

    def _log_stream_response(
        self, request: ChatRequest, start_t: float, response: ChatResponse | None
    ) -> None:
        """Log streaming response with timing and token counts."""
        usage = response.usage if response else None
        self._lg.debug(
            "LLM chat response (streaming)",
            extra={
                "after": since(start_t),
                "req": request.id,
                "model": request.model,
                "backend": self._backend.name,
                "tokens": _tokens_log(usage),
            },
        )

    def _chat(self, request: ChatRequest) -> ChatResponse:
        """Internal: send chat request (sync) with retry."""
        self._lg.debug(
            "LLM chat request...",
            extra={
                "req": request.id,
                "model": request.model,
                "backend": self._backend.name,
            },
        )
        start_t = start()
        response = self._retry.call(
            lambda: self._backend.chat(request),
            request=request,
            callbacks=self._callbacks,
        )
        usage = response.usage
        self._lg.debug(
            "LLM chat response",
            extra={
                "after": since(start_t),
                "req": request.id,
                "model": request.model,
                "backend": self._backend.name,
                "tokens": _tokens_log(usage),
            },
        )
        return response

    def _chat_stream(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> Iterator[str]:
        """Internal: stream chat request (sync).

        Transient errors are retried with backoff until the first token is
        yielded (same policy as non-streaming); after that, errors propagate.
        on_request fires before streaming (and again per retry attempt);
        on_response and on_error fire after stream completes.
        """
        self._lg.debug(
            "LLM chat request (streaming)...",
            extra={
                "req": request.id,
                "model": request.model,
                "backend": self._backend.name,
            },
        )
        start_t = start()
        self._fire_on_request(request)
        try:
            yield from self._retry.stream(
                lambda: self._backend.chat_stream(request, holder),
                request=request,
                callbacks=self._callbacks,
            )
            response = self._get_stream_response(holder)
            self._log_stream_response(request, start_t, response)
            if response:
                self._fire_on_response(request, response)
        except Exception as e:
            self._fire_on_error(request, e)
            raise

    async def _chat_async(self, request: ChatRequest) -> ChatResponse:
        """Internal: send chat request (async) with retry."""
        self._lg.debug(
            "LLM chat request...",
            extra={
                "req": request.id,
                "model": request.model,
                "backend": self._backend.name,
            },
        )
        start_t = start()
        response = await self._retry.call_async(
            lambda: self._backend.chat_async(request),
            request=request,
            callbacks=self._callbacks,
        )
        usage = response.usage
        self._lg.debug(
            "LLM chat response",
            extra={
                "after": since(start_t),
                "req": request.id,
                "model": request.model,
                "backend": self._backend.name,
                "tokens": _tokens_log(usage),
            },
        )
        return response

    async def _chat_stream_async(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> AsyncIterator[str]:
        """Internal: stream chat request (async).

        Transient errors are retried with backoff until the first token is
        yielded (same policy as non-streaming); after that, errors propagate.
        on_request fires before streaming (and again per retry attempt);
        on_response and on_error fire after stream completes.
        """
        self._lg.debug(
            "LLM chat request (streaming)...",
            extra={
                "req": request.id,
                "model": request.model,
                "backend": self._backend.name,
            },
        )
        start_t = start()
        self._fire_on_request(request)
        try:
            async for token in self._retry.stream_async(
                lambda: self._backend.chat_stream_async(request, holder),
                request=request,
                callbacks=self._callbacks,
            ):
                yield token
            response = self._get_stream_response(holder)
            self._log_stream_response(request, start_t, response)
            if response:
                self._fire_on_response(request, response)
        except Exception as e:
            self._fire_on_error(request, e)
            raise

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        """Close sync resources."""
        self._backend.close()

    async def aclose(self) -> None:
        """Close all resources (sync and async)."""
        await self._backend.aclose()

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
