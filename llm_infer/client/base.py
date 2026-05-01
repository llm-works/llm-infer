"""Abstract base class for chat clients.

This module provides the ChatClient ABC that defines the common interface
for both LLMClient (single-backend) and LLMRouter (multi-backend).

Use ChatClient as a type hint when your code works with either:
    def run_inference(client: ChatClient, messages: list[dict]) -> str:
        return client.chat(messages).content
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any, Self

from .types import ChatResponse


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
    # State
    # =========================================================================

    @property
    @abstractmethod
    def last_response(self) -> ChatResponse | None:
        """The most recent ChatResponse from a completed request.

        Available after chat(), chat_async(), or after fully consuming
        chat_stream()/chat_stream_async(). Returns None if no request
        has completed yet.
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
