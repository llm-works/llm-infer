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

from llm_infer.client.backends import Backend
from llm_infer.client.exceptions import BackendRequestError, BackendUnavailableError
from llm_infer.client.types import ChatResponse

# Transient HTTP status codes that should trigger retry
TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 529})

T = TypeVar("T")


class LLMClient:
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
            rate_limiter: Optional rate limiter for throttling requests.
            backoff: Optional backoff for retrying transient errors. Retries on
                connection failures and HTTP 429/502/503/529 with exponential delay.
            timeout: Total timeout in seconds for retry attempts. 0 = retry forever.
                Only used when backoff is configured.
        """
        self._lg = lg
        self._backend = backend
        self._default_model = default_model
        self._rate_limiter = rate_limiter
        self._backoff = backoff
        self._timeout = timeout

    @property
    def backend(self) -> Backend:
        """The underlying backend instance."""
        return self._backend

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats.

        This is populated after both streaming and non-streaming requests,
        providing access to usage statistics and metadata.
        """
        return self._backend.last_response

    def can_call(self) -> bool:
        """Check if a call is allowed (non-blocking).

        Returns False if rate limited (exceeded per_minute).

        This is an informational check that doesn't consume rate limit slots.
        Use this in event loops to skip cycles when the backend is unavailable.

        Returns:
            True if a call is allowed, False otherwise.
        """
        if self._rate_limiter is not None and not self._rate_limiter.can_proceed():
            return False
        return True

    def _is_transient_error(self, exc: Exception) -> bool:
        """Check if exception is a transient error that should be retried."""
        if isinstance(exc, BackendUnavailableError):
            return True
        if isinstance(exc, BackendRequestError):
            return exc.status_code in TRANSIENT_STATUS_CODES
        return False

    def _call_with_retry(self, fn: Callable[[], T]) -> T:
        """Execute fn with retry on transient errors.

        Retries on connection failures and HTTP 429/502/503/529 with exponential
        backoff. Only active when backoff is configured.

        Args:
            fn: Function to call.

        Returns:
            Result of fn.

        Raises:
            BackendUnavailableError: If connection fails and retries exhausted.
            BackendRequestError: If transient HTTP error and retries exhausted,
                or non-transient error.
        """
        if self._backoff is None:
            return fn()

        self._backoff.reset()
        start_time = time.time()

        while True:
            try:
                result = fn()
                self._backoff.reset()  # Reset on success
                return result
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._is_transient_error(e):
                    raise

                # Check timeout (0 = no timeout)
                elapsed = time.time() - start_time
                if self._timeout > 0 and elapsed >= self._timeout:
                    raise

                delay = self._backoff.next_delay()
                status = e.status_code if isinstance(e, BackendRequestError) else None
                self._lg.warning(
                    "transient error, retrying",
                    extra={"status_code": status, "delay": delay, "elapsed": elapsed},
                )
                time.sleep(delay)

    async def _call_with_retry_async(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute async fn with retry on transient errors.

        Retries on connection failures and HTTP 429/502/503/529 with exponential
        backoff. Only active when backoff is configured.

        Args:
            fn: Async function to call.

        Returns:
            Result of fn.

        Raises:
            BackendUnavailableError: If connection fails and retries exhausted.
            BackendRequestError: If transient HTTP error and retries exhausted,
                or non-transient error.
        """
        if self._backoff is None:
            return await fn()

        self._backoff.reset()
        start_time = time.time()

        while True:
            try:
                result = await fn()
                self._backoff.reset()  # Reset on success
                return result
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._is_transient_error(e):
                    raise

                # Check timeout (0 = no timeout)
                elapsed = time.time() - start_time
                if self._timeout > 0 and elapsed >= self._timeout:
                    raise

                delay = self._backoff.next_delay()
                status = e.status_code if isinstance(e, BackendRequestError) else None
                self._lg.warning(
                    "transient error, retrying",
                    extra={"status_code": status, "delay": delay, "elapsed": elapsed},
                )
                await asyncio.sleep(delay)

    # =========================================================================
    # Sync API
    # =========================================================================

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return content (sync).

        This is the simple API that returns just the generated text.
        For full response with usage stats, use chat_full().

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            Generated text content.
        """
        response = self.chat_full(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        return response.content

    def chat_full(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request and return full response (sync).

        When backoff is configured, automatically retries on transient errors
        (connection failures and HTTP 429/502/503/529) with exponential backoff.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """

        def do_call() -> ChatResponse:
            return self._backend.chat(
                messages=messages,
                model=model or self._default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
                adapter_id=adapter_id,
                think=think,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

        return self._call_with_retry(do_call)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream chat completion tokens (sync).

        Note: Streaming does not support automatic retry. For retry support,
        use chat() or chat_full() instead.

        Yields tokens as they arrive. After iteration, access last_response
        for usage statistics.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        yield from self._backend.chat_stream(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    # =========================================================================
    # Async API
    # =========================================================================

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return content (async).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            Generated text content.
        """
        response = await self.chat_full_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        return response.content

    async def chat_full_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request and return full response (async).

        When backoff is configured, automatically retries on transient errors
        (connection failures and HTTP 429/502/503/529) with exponential backoff.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """

        async def do_call() -> ChatResponse:
            return await self._backend.chat_async(
                messages=messages,
                model=model or self._default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system,
                adapter_id=adapter_id,
                think=think,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

        return await self._call_with_retry_async(do_call)

    async def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens (async).

        Note: Streaming does not support automatic retry. For retry support,
        use chat_async() or chat_full_async() instead.

        Yields tokens as they arrive. After iteration, access last_response
        for usage statistics.

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        async for token in self._backend.chat_stream_async(
            messages=messages,
            model=model or self._default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
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
