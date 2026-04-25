"""Abstract base class for LLM backends.

All backend implementations must inherit from Backend and implement
both sync and async methods for chat completion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from appinfra.log import Logger

from ..types import ChatRequest, ChatResponse
from .context import BackendContext


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

    def __init__(
        self,
        lg: Logger,
        name: str,
        ctx: BackendContext | None = None,
        default_model: str | None = None,
    ) -> None:
        """Initialize backend with common configuration.

        Args:
            lg: Logger instance.
            name: Backend name (used for discovery/routing).
            ctx: Backend context with rate limiter and backoff config.
            default_model: Default model to use if not specified per-request.
        """
        self._lg = lg
        self._name = name
        self._ctx = ctx or BackendContext()
        self._default_model = default_model

    @property
    def name(self) -> str:
        """Backend name (used for discovery/routing)."""
        return self._name

    @property
    def default_model(self) -> str | None:
        """Default model used when not specified per-request."""
        return self._default_model

    @property
    def ctx(self) -> BackendContext:
        """Backend context with rate limiter and backoff."""
        return self._ctx

    def can_call(self) -> bool:
        """Check if a call would be allowed right now (non-blocking).

        Returns False if rate limited. Use in event loops to skip cycles
        when the rate limit would block.

        Returns:
            True if a call would be allowed, False otherwise.
        """
        if (
            self._ctx.rate_limiter is not None
            and not self._ctx.rate_limiter.can_proceed()
        ):
            return False
        return True

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
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming chat completion request (sync).

        Args:
            request: Chat request with messages, model, and parameters.

        Returns:
            ChatResponse with content, usage, and optional extensions.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    @abstractmethod
    def chat_stream(self, request: ChatRequest) -> Iterator[str]:
        """Send a streaming chat completion request (sync).

        Yields tokens as they arrive. After iteration completes, access
        `last_response` for usage statistics and metadata.

        Args:
            request: Chat request with messages, model, and parameters.

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
    async def chat_async(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming chat completion request (async).

        Args:
            request: Chat request with messages, model, and parameters.

        Returns:
            ChatResponse with content, usage, and optional extensions.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    @abstractmethod
    def chat_stream_async(self, request: ChatRequest) -> AsyncIterator[str]:
        """Send a streaming chat completion request (async).

        Yields tokens as they arrive. After iteration completes, access
        `last_response` for usage statistics and metadata.

        Note: This method is an async generator. Call it without await:
            async for token in backend.chat_stream_async(request):
                ...

        Args:
            request: Chat request with messages, model, and parameters.

        Yields:
            String tokens as they arrive.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    # =========================================================================
    # Model discovery
    # =========================================================================

    def list_models(self) -> list[str]:
        """List available models from this backend.

        Returns a list of model IDs that this backend can serve. Used by
        LLMRouter for intelligent model-based routing.

        Default implementation returns empty list. Backends that support
        model discovery (e.g., OpenAI-compatible via /v1/models) should
        override this method.

        Returns:
            List of model ID strings.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendRequestError: Backend returned an error.
        """
        return []

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
