"""Embedding client with retry support.

Wraps an embedding backend with retry logic for transient errors.

Usage:
    from appinfra.log import Logger
    from llm_infer.client.backends import embedding

    lg = Logger("my-app")

    # Create backend and use directly (no retry)
    backend = embedding.OpenAIBackend(
        lg, base_url="https://api.openai.com/v1",
        api_key="sk-...", model="text-embedding-3-small"
    )
    result = backend.embed("Hello world")

    # Or wrap with client for retry support
    from llm_infer.client import EmbeddingClient
    from llm_infer.client.backends import RetryConfig

    client = EmbeddingClient(lg, backend, retry=RetryConfig(timeout=120.0))
    result = client.embed("Hello world")

    # Google embeddings
    backend = embedding.GoogleBackend(
        lg, api_key="AIza...", model="gemini-embedding-001",
        task_type=embedding.TaskType.RETRIEVAL_DOCUMENT
    )
    result = backend.embed("Hello world")
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any, Self, TypeVar

from appinfra.log import Logger

from .backends import RetryConfig
from .backends.embedding import Backend, BatchEmbeddingResult, EmbeddingResult
from .errors import BackendRequestError, BackendUnavailableError
from .retry import RetryBase

T = TypeVar("T")


class EmbeddingClient:
    """Embedding client with retry support.

    Wraps an embedding backend and adds retry logic for transient errors
    (5xx, 429, 529).

    For simple usage without retry, use the backend directly.
    """

    def __init__(
        self,
        lg: Logger,
        backend: Backend,
        retry: RetryConfig | None = None,
    ) -> None:
        """Initialize the embedding client.

        Args:
            lg: Logger for retry/error logging.
            backend: Embedding backend to use.
            retry: Retry configuration for transient errors. None disables retry.
        """
        self._lg = lg
        self._backend = backend
        self._retry = retry
        self._retry_base = RetryBase(lg)

    @property
    def model(self) -> str:
        """Model name from the backend."""
        return self._backend.model

    @property
    def backend(self) -> Backend:
        """The underlying embedding backend."""
        return self._backend

    # =========================================================================
    # Retry wrappers
    # =========================================================================

    def _call_with_retry(self, func: Callable[[], T]) -> T:
        """Execute function with retry on transient errors."""
        if self._retry is None:
            return func()

        backoff = self._retry_base.create_backoff(self._retry)
        start_time = time.monotonic()
        retry_count = 0
        while True:
            try:
                return func()
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._retry_base.should_retry(
                    e, start_time, self._retry.timeout
                ):
                    raise
                delay = self._retry_base.compute_delay(
                    backoff, self._retry.timeout, start_time
                )
                if delay is None:
                    raise
                retry_count += 1
                self._lg.warning(
                    "embedding request failed, retrying",
                    extra={"retry": retry_count, "delay": delay, "exception": e},
                )
                time.sleep(delay)

    async def _call_with_retry_async(
        self, coro_func: Callable[[], Coroutine[Any, Any, T]]
    ) -> T:
        """Execute async function with retry on transient errors."""
        if self._retry is None:
            return await coro_func()

        backoff = self._retry_base.create_backoff(self._retry)
        start_time = time.monotonic()
        retry_count = 0
        while True:
            try:
                return await coro_func()
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._retry_base.should_retry(
                    e, start_time, self._retry.timeout
                ):
                    raise
                delay = self._retry_base.compute_delay(
                    backoff, self._retry.timeout, start_time
                )
                if delay is None:
                    raise
                retry_count += 1
                self._lg.warning(
                    "embedding request failed, retrying",
                    extra={"retry": retry_count, "delay": delay, "exception": e},
                )
                await asyncio.sleep(delay)

    # =========================================================================
    # Sync API
    # =========================================================================

    def embed(self, text: str, *, dimensions: int | None = None) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.
            dimensions: Output dimensions. None uses model default.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            BackendUnavailableError: If the backend is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the backend returns an error.
        """
        return self._call_with_retry(
            lambda: self._backend.embed(text, dimensions=dimensions)
        )

    def embed_batch(
        self, texts: list[str], *, dimensions: int | None = None
    ) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            dimensions: Output dimensions. None uses model default.

        Returns:
            BatchEmbeddingResult with results list and total_prompt_tokens.

        Raises:
            BackendUnavailableError: If the backend is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the backend returns an error.
        """
        if not texts:
            return BatchEmbeddingResult(results=[], total_prompt_tokens=0)
        return self._call_with_retry(
            lambda: self._backend.embed_batch(texts, dimensions=dimensions)
        )

    # =========================================================================
    # Async API
    # =========================================================================

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
            BackendUnavailableError: If the backend is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the backend returns an error.
        """
        return await self._call_with_retry_async(
            lambda: self._backend.embed_async(text, dimensions=dimensions)
        )

    async def embed_batch_async(
        self, texts: list[str], *, dimensions: int | None = None
    ) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts (async).

        Args:
            texts: List of texts to embed.
            dimensions: Output dimensions. None uses model default.

        Returns:
            BatchEmbeddingResult with results list and total_prompt_tokens.

        Raises:
            BackendUnavailableError: If the backend is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the backend returns an error.
        """
        if not texts:
            return BatchEmbeddingResult(results=[], total_prompt_tokens=0)
        return await self._call_with_retry_async(
            lambda: self._backend.embed_batch_async(texts, dimensions=dimensions)
        )

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        """Close the backend's sync resources."""
        self._backend.close()

    async def aclose(self) -> None:
        """Close all backend resources (sync and async)."""
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
