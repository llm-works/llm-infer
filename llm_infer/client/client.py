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

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Self, TypeVar

from appinfra.log import Logger
from appinfra.rate_limit import Backoff, RateLimiter

from .backends import Backend
from .base import ChatClient
from .errors import BackendRequestError, BackendUnavailableError
from .types import ChatResponse

# Non-5xx status codes that should trigger retry (5xx are always retried)
# 429 = rate limited, 529 = site overloaded (Cloudflare)
TRANSIENT_4XX_CODES: frozenset[int] = frozenset({429, 529})

T = TypeVar("T")


class LLMClient(ChatClient):
    """Single-backend LLM client with sync/async support.

    This client wraps a single backend and provides a consistent interface
    for chat completions. For multi-backend routing, use LLMRouter.

    The client delegates all operations to an underlying Backend instance,
    which handles the actual API communication.

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
        default_model: str | None = None,
        rate_limiter: RateLimiter | None = None,
        backoff: Backoff | None = None,
        timeout: float = 0,
    ) -> None:
        """Initialize the client with a backend.

        Args:
            lg: Logger instance.
            backend: The backend to use for API calls.
            default_model: Default model to use if not specified per-request.
            rate_limiter: Optional rate limiter for throttling requests. When set,
                blocks before each request until the rate limit allows. This caps
                the maximum request rate regardless of errors or retries.
            backoff: Optional backoff for retrying transient errors. Retries on
                connection failures and HTTP 5xx/429/529 errors with exponential delay.
                Also acts as a gatekeeper, applying cooldown after any error to slow
                down subsequent requests. For production use, configure both rate_limiter
                (caps steady-state rate) and backoff (slows down during errors).
            timeout: Total timeout in seconds for retry attempts. 0 = retry forever.
                Only used when backoff is configured.
        """
        self._lg = lg
        self._backend = backend
        self._default_model = default_model
        self._rate_limiter = rate_limiter
        self._backoff = backoff
        self._timeout = timeout

        # Inject rate limiter into backend for deep rate limiting
        backend.set_rate_limiter(rate_limiter)

    @property
    def backend(self) -> Backend:
        """The underlying backend instance."""
        return self._backend

    @property
    def default_model(self) -> str | None:
        """Default model used when not specified per-request."""
        return self._default_model

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats.

        This is populated after both streaming and non-streaming requests,
        providing access to usage statistics and metadata.
        """
        return self._backend.last_response

    def can_call(self) -> bool:
        """Check if a call would be allowed right now (non-blocking).

        Returns False if rate limited (exceeded per_minute).

        Note: Rate limiting is enforced automatically on each request, so
        calling this method is optional. Use this in event loops to skip
        cycles when the rate limit would block.

        Returns:
            True if a call would be allowed, False otherwise.
        """
        if self._rate_limiter is not None and not self._rate_limiter.can_proceed():
            return False
        return True

    def _is_transient_error(self, exc: Exception) -> bool:
        """Check if exception is a transient error that should be retried.

        Retries on:
        - BackendUnavailableError (connection failures)
        - Transport errors (no status code - connection dropped mid-request)
        - All 5xx server errors (500-599)
        - 429 (rate limited) and 529 (Cloudflare overloaded)

        Does NOT retry:
        - 4xx client errors (except 429/529) - these indicate bad requests
        """
        if isinstance(exc, BackendUnavailableError):
            return True
        if isinstance(exc, BackendRequestError):
            code = exc.status_code
            if code is None:
                # Transport error (connection dropped, stale connection, etc.)
                return True
            # All 5xx server errors are retryable
            if 500 <= code < 600:
                return True
            # Specific 4xx codes that are retryable
            return code in TRANSIENT_4XX_CODES
        return False

    def _apply_backoff_cooldown(self) -> None:
        """Apply cooldown delay if previous calls have failed.

        This acts as a gatekeeper, preventing rapid-fire requests when errors
        occur - even for non-transient errors like 400 Bad Request.
        """
        if self._backoff is None or self._backoff.attempts == 0:
            return
        # Calculate delay without incrementing (next_delay would increment)
        delay = min(
            self._backoff.base * (self._backoff.factor ** (self._backoff.attempts - 1)),
            self._backoff.max_delay,
        )
        self._lg.debug(
            "backoff cooldown before request",
            extra={"delay": delay, "attempts": self._backoff.attempts},
        )
        time.sleep(delay)

    async def _apply_backoff_cooldown_async(self) -> None:
        """Async version of _apply_backoff_cooldown."""
        if self._backoff is None or self._backoff.attempts == 0:
            return
        delay = min(
            self._backoff.base * (self._backoff.factor ** (self._backoff.attempts - 1)),
            self._backoff.max_delay,
        )
        self._lg.debug(
            "backoff cooldown before request",
            extra={"delay": delay, "attempts": self._backoff.attempts},
        )
        await asyncio.sleep(delay)

    def _handle_retry_error(
        self, e: BackendUnavailableError | BackendRequestError, start_time: float
    ) -> float:
        """Handle error during retry, returning delay if should retry, or re-raising."""
        assert self._backoff is not None  # Only called when backoff is configured
        if not self._is_transient_error(e):
            self._backoff.next_delay()  # Increment for next call
            raise
        elapsed = time.time() - start_time
        if self._timeout > 0 and elapsed >= self._timeout:
            self._backoff.next_delay()
            raise
        delay: float = self._backoff.next_delay()
        status = e.status_code if isinstance(e, BackendRequestError) else None
        self._lg.warning(
            "transient error, retrying",
            extra={"status_code": status, "delay": delay, "elapsed": elapsed},
        )
        return delay

    def _call_with_retry(self, fn: Callable[[], T]) -> T:
        """Execute fn with retry on transient errors and gatekeeper cooldown.

        Raises:
            BackendUnavailableError: If connection fails and retries exhausted.
            BackendRequestError: If HTTP error occurs and retries exhausted,
                or if error is non-transient (e.g., 400 Bad Request).
        """
        # Enforce rate limit (blocks until allowed)
        if self._rate_limiter is not None:
            self._rate_limiter.next()

        if self._backoff is None:
            return fn()
        self._apply_backoff_cooldown()
        start_time = time.time()
        while True:
            try:
                result = fn()
                self._backoff.reset()
                return result
            except (BackendUnavailableError, BackendRequestError) as e:
                delay = self._handle_retry_error(e, start_time)
                time.sleep(delay)

    async def _call_with_retry_async(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute async fn with retry on transient errors and gatekeeper cooldown.

        Raises:
            BackendUnavailableError: If connection fails and retries exhausted.
            BackendRequestError: If HTTP error occurs and retries exhausted,
                or if error is non-transient (e.g., 400 Bad Request).
        """
        # Enforce rate limit (run blocking call in thread to not block event loop)
        if self._rate_limiter is not None:
            await asyncio.to_thread(self._rate_limiter.next)

        if self._backoff is None:
            return await fn()
        await self._apply_backoff_cooldown_async()
        start_time = time.time()
        while True:
            try:
                result = await fn()
                self._backoff.reset()
                return result
            except (BackendUnavailableError, BackendRequestError) as e:
                delay = self._handle_retry_error(e, start_time)
                await asyncio.sleep(delay)

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
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """

        def do_call() -> ChatResponse:
            return self._backend.chat(
                messages=messages,
                model=model or self._default_model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                think=think,
                adapter=adapter,
                **kwargs,
            )

        return self._call_with_retry(do_call)

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
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream chat completion tokens (sync).

        Note: Streaming does not support automatic retry. For retry support,
        use chat() instead.

        Yields tokens as they arrive. After iteration, access last_response
        for usage statistics.

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
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        # Enforce rate limit (blocks until allowed)
        if self._rate_limiter is not None:
            self._rate_limiter.next()

        yield from self._backend.chat_stream(
            messages=messages,
            model=model or self._default_model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            **kwargs,
        )

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
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """

        async def do_call() -> ChatResponse:
            return await self._backend.chat_async(
                messages=messages,
                model=model or self._default_model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                think=think,
                adapter=adapter,
                **kwargs,
            )

        return await self._call_with_retry_async(do_call)

    async def chat_stream_async(
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
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens (async).

        Note: Streaming does not support automatic retry. For retry support,
        use chat_async() instead.

        Yields tokens as they arrive. After iteration, access last_response
        for usage statistics.

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
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        # Enforce rate limit (run blocking call in thread to not block event loop)
        if self._rate_limiter is not None:
            await asyncio.to_thread(self._rate_limiter.next)

        async for token in self._backend.chat_stream_async(
            messages=messages,
            model=model or self._default_model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
            **kwargs,
        ):
            yield token

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
