"""Embedding backend implementations.

Provides backend base class and provider implementations for embedding generation.

Usage:
    from llm_infer.client.backends import embedding

    # OpenAI-compatible backend
    backend = embedding.OpenAIBackend(
        lg=logger,
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="text-embedding-3-small",
    )

    # Google backend
    from llm_infer.client.backends.providers.google import GoogleEmbeddingTaskType

    backend = embedding.GoogleBackend(
        lg=logger,
        api_key="AIza...",
        model="text-embedding-004",
        task_type=GoogleEmbeddingTaskType.RETRIEVAL_DOCUMENT,
    )

    result = backend.embed("Hello world")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

from appinfra.log import Logger


@dataclass
class EmbeddingResult:
    """Result from embedding generation.

    Attributes:
        embedding: The embedding vector.
        model: Model name that generated the embedding.
        dimensions: Number of dimensions in the embedding vector.
        prompt_tokens: Number of tokens in the input. None if the backend
            doesn't report token counts (use backend.count_tokens() to get it).
    """

    embedding: list[float]
    model: str
    dimensions: int
    prompt_tokens: int | None


class Backend(ABC):
    """Abstract base class for embedding backends.

    All backends must implement both synchronous and asynchronous methods
    for embedding generation. Backends handle connection management internally
    and translate backend-specific errors to the BackendError hierarchy.

    Example:
        # Sync usage
        with SomeBackend(...) as backend:
            result = backend.embed("Hello world")

        # Async usage
        async with SomeBackend(...) as backend:
            result = await backend.embed_async("Hello world")
    """

    def __init__(self, lg: Logger, model: str) -> None:
        """Initialize backend with common configuration.

        Args:
            lg: Logger instance.
            model: Model name to use for embeddings.
        """
        self._lg = lg
        self._model = model

    @property
    def model(self) -> str:
        """Model name used for embeddings."""
        return self._model

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider identifier for this backend (e.g., 'openai', 'google')."""
        ...

    # =========================================================================
    # Sync methods
    # =========================================================================

    @abstractmethod
    def embed(self, text: str, *, dimensions: int | None = None) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.
            dimensions: Output dimensions. None uses model default.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    @abstractmethod
    def embed_batch(
        self, texts: list[str], *, dimensions: int | None = None
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            dimensions: Output dimensions. None uses model default.

        Returns:
            List of EmbeddingResult, one per input text.

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
    async def embed_async(
        self, text: str, *, dimensions: int | None = None
    ) -> EmbeddingResult:
        """Generate embedding for a single text (async).

        Args:
            text: Text to embed.
            dimensions: Output dimensions. None uses model default.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    @abstractmethod
    async def embed_batch_async(
        self, texts: list[str], *, dimensions: int | None = None
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts (async).

        Args:
            texts: List of texts to embed.
            dimensions: Output dimensions. None uses model default.

        Returns:
            List of EmbeddingResult, one per input text.

        Raises:
            BackendUnavailableError: Backend is unreachable.
            BackendTimeoutError: Request timed out.
            BackendRequestError: Backend returned an error.
        """
        ...

    # =========================================================================
    # Token counting
    # =========================================================================

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for a single text.

        For backends that return prompt_tokens in EmbeddingResult (e.g., OpenAI),
        this uses a local tokenizer. For backends that don't (e.g., Google),
        this may require an API call.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        ...

    @abstractmethod
    def count_tokens_batch(self, texts: list[str]) -> int:
        """Count total tokens for multiple texts.

        Args:
            texts: List of texts to count tokens for.

        Returns:
            Total number of tokens across all texts.
        """
        ...

    @abstractmethod
    async def count_tokens_async(self, text: str) -> int:
        """Count tokens for a single text (async).

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        ...

    @abstractmethod
    async def count_tokens_batch_async(self, texts: list[str]) -> int:
        """Count total tokens for multiple texts (async).

        Args:
            texts: List of texts to count tokens for.

        Returns:
            Total number of tokens across all texts.
        """
        ...

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        """Close sync resources.

        Override to close sync HTTP clients. Called by __exit__.
        """
        pass

    async def aclose(self) -> None:
        """Close all resources (sync and async).

        Override to close both sync and async HTTP clients.
        Called by __aexit__.
        """
        pass

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


# =============================================================================
# Provider re-exports with shorter names
# =============================================================================


def __getattr__(name: str) -> type:
    """Lazy import provider backends to avoid circular imports."""
    if name == "GoogleBackend":
        from .providers.google import GoogleEmbeddingBackend

        return GoogleEmbeddingBackend
    if name == "OpenAIBackend":
        from .providers.openai import OpenAIEmbeddingBackend

        return OpenAIEmbeddingBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Backend",
    "EmbeddingResult",
]
