"""Abstract base class for LLM backends.

All backend implementations must inherit from Backend and implement
both sync and async methods for chat completion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from llm_infer.client.types import ChatResponse


class Backend(ABC):
    """Abstract base class for LLM backends.

    All backends must implement both synchronous and asynchronous methods
    for chat completion. Backends handle connection management internally
    and translate backend-specific errors to the BackendError hierarchy.

    Resource Management:
        Backends should create sync HTTP clients eagerly in __init__ and
        async clients lazily on first async call. Use close() for sync
        cleanup and aclose() for async cleanup (which handles both).

    Example:
        # Sync usage
        with SomeBackend(...) as backend:
            response = backend.chat(messages)

        # Async usage
        async with SomeBackend(...) as backend:
            response = await backend.chat_async(messages)
    """

    @property
    @abstractmethod
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats.

        This is populated after both streaming and non-streaming requests,
        providing access to usage statistics and metadata that may only
        be available at the end of generation.
        """
        ...

    # =========================================================================
    # Sync methods
    # =========================================================================

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        # llm-infer extensions
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat completion request (sync).

        Args:
            messages: List of chat messages as dicts with 'role' and 'content'.
            model: Model to use (overrides default if set).
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            system: System prompt (prepended to messages or passed separately).
            adapter_id: LoRA adapter name (llm-infer extension, OpenAI only).
            think: Enable thinking mode (llm-infer extension).
            tools: List of tool definitions for function calling.
            tool_choice: Control tool use ('auto', 'none', 'required', or specific).
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, and optional extensions.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        # llm-infer extensions
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Send a streaming chat completion request (sync).

        Yields tokens as they arrive. After iteration completes, access
        `last_response` for usage statistics and metadata.

        Args:
            messages: List of chat messages.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (llm-infer extension).
            think: Enable thinking mode (llm-infer extension).
            tools: List of tool definitions.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    # =========================================================================
    # Async methods
    # =========================================================================

    @abstractmethod
    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        # llm-infer extensions
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat completion request (async).

        Args:
            messages: List of chat messages.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (llm-infer extension).
            think: Enable thinking mode (llm-infer extension).
            tools: List of tool definitions.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, and optional extensions.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    @abstractmethod
    def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        # llm-infer extensions
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request (async).

        Yields tokens as they arrive. After iteration completes, access
        `last_response` for usage statistics and metadata.

        Note: This method is an async generator. Call it without await:
            async for token in backend.chat_stream_async(messages):
                ...

        Args:
            messages: List of chat messages.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (llm-infer extension).
            think: Enable thinking mode (llm-infer extension).
            tools: List of tool definitions.
            tool_choice: Control tool use.
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        """Close sync resources.

        Override this method to close sync HTTP clients and other resources.
        Called by __exit__.
        """
        pass

    async def aclose(self) -> None:
        """Close all resources (sync and async).

        Override this method to close both sync and async HTTP clients.
        Called by __aexit__. Should also close sync resources since async
        context manager exit is the final cleanup opportunity.
        """
        pass

    def __enter__(self) -> Backend:
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

    async def __aenter__(self) -> Backend:
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

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Backend:
        """Create a backend from configuration dict.

        Args:
            config: Configuration dictionary with backend-specific settings.

        Returns:
            Configured backend instance.
        """
        ...
