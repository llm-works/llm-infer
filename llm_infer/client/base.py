"""Abstract base class for chat clients.

This module provides the ChatClient ABC that defines the common interface
for both LLMClient (single-backend) and LLMRouter (multi-backend).

Use ChatClient as a type hint when your code works with either:
    def run_inference(client: ChatClient, messages: list[dict]) -> str:
        return client.chat(messages).content

Also provides BoundChatClient for binding kwargs to any ChatClient:
    bound = BoundChatClient(router, role="exploration")
    bound.chat(messages)  # role="exploration" merged automatically
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from enum import Enum, auto
from typing import Any, Self

from .types import ChatResponse


class _Unset(Enum):
    """Sentinel for detecting unset parameters in BoundChatClient."""

    UNSET = auto()


_UNSET = _Unset.UNSET


class ChatClient(ABC):
    """Abstract base class for chat completion clients.

    This ABC defines the common interface shared by LLMClient and LLMRouter.
    Use this type when your code should work with either implementation.

    Both sync and async APIs are provided:
        - chat() / chat_async() - Full response with metadata
        - chat_stream() / chat_stream_async() - Streaming tokens

    Resource management:
        - Use as context manager (sync or async)
        - Or call close() / aclose() explicitly
    """

    # =========================================================================
    # Rate limiting
    # =========================================================================

    @abstractmethod
    def can_call(self) -> bool:
        """Check if a call is allowed (non-blocking).

        Returns:
            True if a call is allowed, False if rate limited or in backoff.
        """
        ...

    # =========================================================================
    # Sync API
    # =========================================================================

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    # =========================================================================
    # Async API
    # =========================================================================

    @abstractmethod
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
        ...

    @abstractmethod
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
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens (async).

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
        ...

    # =========================================================================
    # Resource management
    # =========================================================================

    @abstractmethod
    def close(self) -> None:
        """Close sync resources."""
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Close async resources."""
        ...

    @abstractmethod
    def __enter__(self) -> Self:
        """Enter sync context manager."""
        ...

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context manager."""
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        ...


class BoundChatClient(ChatClient):
    """ChatClient wrapper that binds kwargs to every call.

    Creates a view of a ChatClient with pre-bound arguments that are merged
    into every chat call. Useful for binding routing parameters (role, backend)
    without modifying the underlying client.

    The bound client delegates resource management to the wrapped client.
    Closing the bound client closes the underlying client.

    Example:
        router = Factory(lg).from_config(config)
        exploration = BoundChatClient(router, role="exploration")
        synthesis = BoundChatClient(router, role="synthesis")

        # Both use the same router, different roles
        exploration.chat(messages)  # role="exploration" merged
        synthesis.chat(messages)    # role="synthesis" merged
    """

    def __init__(self, client: ChatClient, **kwargs: Any) -> None:
        """Create a bound view of a ChatClient.

        Args:
            client: The ChatClient to wrap.
            **kwargs: Arguments to merge into every chat call.
        """
        self._client = client
        self._bound_kwargs = kwargs

    @property
    def client(self) -> ChatClient:
        """The underlying ChatClient."""
        return self._client

    @property
    def bound_kwargs(self) -> dict[str, Any]:
        """The bound kwargs (read-only copy)."""
        return dict(self._bound_kwargs)

    def with_chat_args(self, **kwargs: Any) -> BoundChatClient:
        """Create a new BoundChatClient with additional bound kwargs.

        Args:
            **kwargs: Additional arguments to merge.

        Returns:
            New BoundChatClient with merged kwargs.
        """
        merged = {**self._bound_kwargs, **kwargs}
        return BoundChatClient(self._client, **merged)

    def can_call(self) -> bool:
        """Check if a call is allowed (delegates to wrapped client)."""
        return self._client.can_call()

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset = _UNSET,
        system: str | None | _Unset = _UNSET,
        temperature: float | _Unset = _UNSET,
        max_tokens: int | None | _Unset = _UNSET,
        tools: list[dict[str, Any]] | None | _Unset = _UNSET,
        tool_choice: str | dict[str, Any] | None | _Unset = _UNSET,
        think: bool | None | _Unset = _UNSET,
        adapter: str | None | _Unset = _UNSET,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request with bound kwargs merged."""
        call_kwargs: dict[str, Any] = {**self._bound_kwargs, "messages": messages}
        if model is not _UNSET:
            call_kwargs["model"] = model
        if system is not _UNSET:
            call_kwargs["system"] = system
        if temperature is not _UNSET:
            call_kwargs["temperature"] = temperature
        if max_tokens is not _UNSET:
            call_kwargs["max_tokens"] = max_tokens
        if tools is not _UNSET:
            call_kwargs["tools"] = tools
        if tool_choice is not _UNSET:
            call_kwargs["tool_choice"] = tool_choice
        if think is not _UNSET:
            call_kwargs["think"] = think
        if adapter is not _UNSET:
            call_kwargs["adapter"] = adapter
        call_kwargs.update(kwargs)
        return self._client.chat(**call_kwargs)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset = _UNSET,
        system: str | None | _Unset = _UNSET,
        temperature: float | _Unset = _UNSET,
        max_tokens: int | None | _Unset = _UNSET,
        tools: list[dict[str, Any]] | None | _Unset = _UNSET,
        tool_choice: str | dict[str, Any] | None | _Unset = _UNSET,
        think: bool | None | _Unset = _UNSET,
        adapter: str | None | _Unset = _UNSET,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream chat completion tokens with bound kwargs merged."""
        call_kwargs: dict[str, Any] = {**self._bound_kwargs, "messages": messages}
        if model is not _UNSET:
            call_kwargs["model"] = model
        if system is not _UNSET:
            call_kwargs["system"] = system
        if temperature is not _UNSET:
            call_kwargs["temperature"] = temperature
        if max_tokens is not _UNSET:
            call_kwargs["max_tokens"] = max_tokens
        if tools is not _UNSET:
            call_kwargs["tools"] = tools
        if tool_choice is not _UNSET:
            call_kwargs["tool_choice"] = tool_choice
        if think is not _UNSET:
            call_kwargs["think"] = think
        if adapter is not _UNSET:
            call_kwargs["adapter"] = adapter
        call_kwargs.update(kwargs)
        yield from self._client.chat_stream(**call_kwargs)

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset = _UNSET,
        system: str | None | _Unset = _UNSET,
        temperature: float | _Unset = _UNSET,
        max_tokens: int | None | _Unset = _UNSET,
        tools: list[dict[str, Any]] | None | _Unset = _UNSET,
        tool_choice: str | dict[str, Any] | None | _Unset = _UNSET,
        think: bool | None | _Unset = _UNSET,
        adapter: str | None | _Unset = _UNSET,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request (async) with bound kwargs merged."""
        call_kwargs: dict[str, Any] = {**self._bound_kwargs, "messages": messages}
        if model is not _UNSET:
            call_kwargs["model"] = model
        if system is not _UNSET:
            call_kwargs["system"] = system
        if temperature is not _UNSET:
            call_kwargs["temperature"] = temperature
        if max_tokens is not _UNSET:
            call_kwargs["max_tokens"] = max_tokens
        if tools is not _UNSET:
            call_kwargs["tools"] = tools
        if tool_choice is not _UNSET:
            call_kwargs["tool_choice"] = tool_choice
        if think is not _UNSET:
            call_kwargs["think"] = think
        if adapter is not _UNSET:
            call_kwargs["adapter"] = adapter
        call_kwargs.update(kwargs)
        return await self._client.chat_async(**call_kwargs)

    async def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None | _Unset = _UNSET,
        system: str | None | _Unset = _UNSET,
        temperature: float | _Unset = _UNSET,
        max_tokens: int | None | _Unset = _UNSET,
        tools: list[dict[str, Any]] | None | _Unset = _UNSET,
        tool_choice: str | dict[str, Any] | None | _Unset = _UNSET,
        think: bool | None | _Unset = _UNSET,
        adapter: str | None | _Unset = _UNSET,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens (async) with bound kwargs merged."""
        call_kwargs: dict[str, Any] = {**self._bound_kwargs, "messages": messages}
        if model is not _UNSET:
            call_kwargs["model"] = model
        if system is not _UNSET:
            call_kwargs["system"] = system
        if temperature is not _UNSET:
            call_kwargs["temperature"] = temperature
        if max_tokens is not _UNSET:
            call_kwargs["max_tokens"] = max_tokens
        if tools is not _UNSET:
            call_kwargs["tools"] = tools
        if tool_choice is not _UNSET:
            call_kwargs["tool_choice"] = tool_choice
        if think is not _UNSET:
            call_kwargs["think"] = think
        if adapter is not _UNSET:
            call_kwargs["adapter"] = adapter
        call_kwargs.update(kwargs)
        async for token in self._client.chat_stream_async(**call_kwargs):
            yield token

    def close(self) -> None:
        """Close the wrapped client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the wrapped client (async)."""
        await self._client.aclose()

    def __enter__(self) -> Self:
        """Enter sync context manager."""
        self._client.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context manager."""
        self._client.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self._client.__aexit__(exc_type, exc_val, exc_tb)
