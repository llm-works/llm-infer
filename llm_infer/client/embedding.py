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
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Self, TypeVar

from appinfra.log import Logger
from appinfra.time import since, start

from .backends import RetryConfig
from .backends.embedding import Backend, BatchEmbeddingResult, EmbeddingResult
from .errors import BackendRequestError, BackendUnavailableError
from .retry import RetryBase
from .types import _gen_req_id


@dataclass
class EmbeddingRequest:
    """Request for an embedding generation.

    Captures the parameters sent to an embedding endpoint. Passed to
    ``EmbeddingCallbacks`` for logging, cost tracking, and correlation.
    Mirrors ``ChatRequest`` in the completion path.

    Attributes:
        texts: Input texts. Always a list — single-text ``embed()`` wraps into
            a one-element list.
        model: Model requested (may be None if using client/backend default).
        dimensions: Requested output dimensions. None uses provider default.
        context: User-provided context passed to callbacks (cost tracking, tracing).
        id: Framework-assigned request ID for log correlation.
    """

    texts: list[str]
    model: str | None = None
    dimensions: int | None = None
    context: dict[str, Any] | None = None
    # Assigned by the framework for log correlation. Do not set manually.
    id: str = field(default_factory=_gen_req_id)


EmbeddingRequestCallback = Callable[["EmbeddingRequest", int], None]
EmbeddingResponseCallback = Callable[
    ["EmbeddingRequest", "EmbeddingResult | BatchEmbeddingResult"], None
]
EmbeddingErrorCallback = Callable[["EmbeddingRequest", Exception], None]


# Constrained: the retry/logging helpers hand this back into on_response,
# which only accepts these two shapes.
T = TypeVar("T", EmbeddingResult, BatchEmbeddingResult)


@dataclass
class EmbeddingCallbacks:
    """Callbacks for embedding request lifecycle events.

    Configure callbacks to observe request/response flow for cost tracking,
    logging, tracing, or metrics collection. Mirrors ``LLMCallbacks`` in the
    completion path, but scoped to embedding lifecycle only (no HTTP-level
    or streaming hooks — embeddings do not stream).

    Callback exceptions are caught and logged at WARNING; they never
    propagate into the calling code.

    Fields:
        on_request: Called before each request attempt. Args: (request, retry).
            retry is 0 for the first attempt, 1+ for retries after transient
            errors.
        on_response: Called after a successful response. Args:
            (request, response). ``response`` is an ``EmbeddingResult`` for
            ``embed``/``embed_async`` and a ``BatchEmbeddingResult`` for
            ``embed_batch``/``embed_batch_async``.
        on_error: Called after a terminal failure. Args: (request, exception).
    """

    on_request: EmbeddingRequestCallback | None = None
    on_response: EmbeddingResponseCallback | None = None
    on_error: EmbeddingErrorCallback | None = None


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
        model: str | None = None,
        dimensions: int | None = None,
        callbacks: EmbeddingCallbacks | None = None,
    ) -> None:
        """Initialize the embedding client.

        Args:
            lg: Logger for retry/error logging.
            backend: Embedding backend to use.
            retry: Retry configuration for transient errors. None disables retry.
            model: Default model override. None uses backend default.
            dimensions: Default output dimensions. None uses backend/provider default.
            callbacks: Optional callbacks for request/response/error lifecycle
                events. See ``EmbeddingCallbacks``. Mirrors the ``callbacks``
                surface on ``LLMClient``.
        """
        self._lg = lg
        self._backend = backend
        self._retry = retry
        self._model = model
        self._dimensions = dimensions
        self._retry_base = RetryBase(lg)
        self._callbacks = callbacks

    @property
    def model(self) -> str:
        """Effective model name (client override or backend default)."""
        return self._model or self._backend.model

    @property
    def dimensions(self) -> int | None:
        """Default output dimensions (None uses provider default)."""
        return self._dimensions

    @property
    def backend(self) -> Backend:
        """The underlying embedding backend."""
        return self._backend

    def with_callbacks(self, callbacks: EmbeddingCallbacks) -> Self:
        """Return a client copy with callbacks configured.

        Callbacks fire on request/response/error events for cost tracking,
        logging, tracing, or metrics collection. Mirrors
        ``LLMClient.with_callbacks``.

        Args:
            callbacks: Callbacks for lifecycle events.

        Returns:
            New client instance with callbacks configured. The underlying
            backend and retry configuration are shared with the original.
        """
        clone = copy(self)
        clone._callbacks = callbacks
        return clone

    # =========================================================================
    # Callback firing helpers
    # =========================================================================

    def _fire_on_request(self, request: EmbeddingRequest, retry: int) -> None:
        """Fire on_request callback with error handling."""
        cb = self._callbacks
        if cb and cb.on_request:
            try:
                cb.on_request(request, retry)
            except Exception as e:
                self._lg.warning("on_request callback failed", extra={"exception": e})

    def _fire_on_response(
        self,
        request: EmbeddingRequest,
        response: EmbeddingResult | BatchEmbeddingResult,
    ) -> None:
        """Fire on_response callback with error handling."""
        cb = self._callbacks
        if cb and cb.on_response:
            try:
                cb.on_response(request, response)
            except Exception as e:
                self._lg.warning("on_response callback failed", extra={"exception": e})

    def _fire_on_error(self, request: EmbeddingRequest, error: Exception) -> None:
        """Fire on_error callback with error handling."""
        cb = self._callbacks
        if cb and cb.on_error:
            try:
                cb.on_error(request, error)
            except Exception as cb_err:
                self._lg.warning(
                    "on_error callback failed", extra={"exception": cb_err}
                )

    # =========================================================================
    # Retry wrappers
    # =========================================================================

    def _call_with_retry(
        self,
        request: EmbeddingRequest,
        func: Callable[[], T],
    ) -> T:
        """Execute function with retry on transient errors.

        Fires ``on_request`` on each retry attempt (the initial ``retry=0``
        fire is the caller's responsibility, done in ``_logged_call``).
        """
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
                self._fire_on_request(request, retry_count)

    async def _call_with_retry_async(
        self,
        request: EmbeddingRequest,
        coro_func: Callable[[], Coroutine[Any, Any, T]],
    ) -> T:
        """Execute async function with retry on transient errors.

        Fires ``on_request`` on each retry attempt.
        """
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
                self._fire_on_request(request, retry_count)

    # =========================================================================
    # Logging helpers
    # =========================================================================

    def _log_request(self, req: str, model: str | None, size: dict[str, int]) -> float:
        """Emit entry log and return start_time for pairing."""
        self._lg.debug(
            "embedding request...",
            extra={
                "req": req,
                "model": model,
                "backend": self._backend.provider,
                **size,
            },
        )
        return start()

    def _log_response(
        self,
        req: str,
        model: str | None,
        size: dict[str, int],
        tokens: int | None,
        t0: float,
    ) -> None:
        """Emit response log paired with the entry log by ``req``."""
        self._lg.debug(
            "embedding response",
            extra={
                "after": since(t0),
                "req": req,
                "model": model,
                "backend": self._backend.provider,
                **size,
                "tokens": tokens,
            },
        )

    def _log_failed(
        self,
        req: str,
        model: str | None,
        size: dict[str, int],
        exc: BaseException,
        t0: float,
    ) -> None:
        """Emit failure log so terminal errors aren't silent after entry log."""
        self._lg.debug(
            "embedding failed",
            extra={
                "after": since(t0),
                "req": req,
                "model": model,
                "backend": self._backend.provider,
                **size,
                "exception": exc,
            },
        )

    def _logged_call(
        self,
        request: EmbeddingRequest,
        size: dict[str, int],
        fn: Callable[[EmbeddingRequest], T],
        get_tokens: Callable[[T], int | None],
    ) -> T:
        """Run fn() with paired debug logging and lifecycle callbacks.

        Fires ``on_request(request, 0)`` before the first attempt,
        ``on_response`` on success, ``on_error`` on terminal failure.
        Retry-attempt ``on_request`` fires (retry=1+) are emitted from within
        the retry helper called by ``fn``.
        """
        t0 = self._log_request(request.id, request.model, size)
        self._fire_on_request(request, 0)
        try:
            result = fn(request)
        except BaseException as e:
            self._log_failed(request.id, request.model, size, e, t0)
            if isinstance(e, Exception):
                self._fire_on_error(request, e)
            raise
        self._log_response(request.id, request.model, size, get_tokens(result), t0)
        self._fire_on_response(request, result)
        return result

    async def _logged_call_async(
        self,
        request: EmbeddingRequest,
        size: dict[str, int],
        coro_fn: Callable[[EmbeddingRequest], Coroutine[Any, Any, T]],
        get_tokens: Callable[[T], int | None],
    ) -> T:
        """Async variant of ``_logged_call`` with lifecycle callbacks."""
        t0 = self._log_request(request.id, request.model, size)
        self._fire_on_request(request, 0)
        try:
            result = await coro_fn(request)
        except BaseException as e:
            self._log_failed(request.id, request.model, size, e, t0)
            if isinstance(e, Exception):
                self._fire_on_error(request, e)
            raise
        self._log_response(request.id, request.model, size, get_tokens(result), t0)
        self._fire_on_response(request, result)
        return result

    # =========================================================================
    # Sync API
    # =========================================================================

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.
            model: Model override. None uses client/backend default.
            dimensions: Output dimensions. None uses client/backend/provider default.
            context: User context passed to callbacks (cost tracking, tracing).
                Rides along on the ``EmbeddingRequest`` delivered to
                ``EmbeddingCallbacks``. See ``LLMClient.chat`` for the
                completion-path shape this mirrors.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            BackendUnavailableError: If the backend is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the backend returns an error.
        """
        effective_model = model or self._model
        effective_dims = dimensions if dimensions is not None else self._dimensions
        request = EmbeddingRequest(
            texts=[text],
            model=effective_model,
            dimensions=effective_dims,
            context=context,
        )
        return self._logged_call(
            request,
            {"chars": len(text)},
            lambda req: self._call_with_retry(
                req,
                lambda: self._backend.embed(
                    text, model=effective_model, dimensions=effective_dims
                ),
            ),
            lambda r: r.prompt_tokens,
        )

    def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            model: Model override. None uses client/backend default.
            dimensions: Output dimensions. None uses client/backend/provider default.
            context: User context passed to callbacks (cost tracking, tracing).
                See ``embed`` for details.

        Returns:
            BatchEmbeddingResult with embeddings and metadata.

        Raises:
            BackendUnavailableError: If the backend is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the backend returns an error.
        """
        effective_model = model or self._model
        effective_dims = dimensions if dimensions is not None else self._dimensions
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=effective_model or self._backend.model,
                dimensions=0,
                size=0,
                total_prompt_tokens=0,
            )
        request = EmbeddingRequest(
            texts=texts,
            model=effective_model,
            dimensions=effective_dims,
            context=context,
        )
        return self._logged_call(
            request,
            {"count": len(texts)},
            lambda req: self._call_with_retry(
                req,
                lambda: self._backend.embed_batch(
                    texts, model=effective_model, dimensions=effective_dims
                ),
            ),
            lambda r: r.total_prompt_tokens,
        )

    # =========================================================================
    # Async API
    # =========================================================================

    async def embed_async(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Generate embedding for a single text (async).

        Args:
            text: Text to embed.
            model: Model override. None uses client/backend default.
            dimensions: Output dimensions. None uses client/backend/provider default.
            context: User context passed to callbacks (cost tracking, tracing).
                See ``embed`` for details.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            BackendUnavailableError: If the backend is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the backend returns an error.
        """
        effective_model = model or self._model
        effective_dims = dimensions if dimensions is not None else self._dimensions
        request = EmbeddingRequest(
            texts=[text],
            model=effective_model,
            dimensions=effective_dims,
            context=context,
        )
        return await self._logged_call_async(
            request,
            {"chars": len(text)},
            lambda req: self._call_with_retry_async(
                req,
                lambda: self._backend.embed_async(
                    text, model=effective_model, dimensions=effective_dims
                ),
            ),
            lambda r: r.prompt_tokens,
        )

    async def embed_batch_async(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts (async).

        Args:
            texts: List of texts to embed.
            model: Model override. None uses client/backend default.
            dimensions: Output dimensions. None uses client/backend/provider default.
            context: User context passed to callbacks (cost tracking, tracing).
                See ``embed`` for details.

        Returns:
            BatchEmbeddingResult with embeddings and metadata.

        Raises:
            BackendUnavailableError: If the backend is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the backend returns an error.
        """
        effective_model = model or self._model
        effective_dims = dimensions if dimensions is not None else self._dimensions
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=effective_model or self._backend.model,
                dimensions=0,
                size=0,
                total_prompt_tokens=0,
            )
        request = EmbeddingRequest(
            texts=texts,
            model=effective_model,
            dimensions=effective_dims,
            context=context,
        )
        return await self._logged_call_async(
            request,
            {"count": len(texts)},
            lambda req: self._call_with_retry_async(
                req,
                lambda: self._backend.embed_batch_async(
                    texts, model=effective_model, dimensions=effective_dims
                ),
            ),
            lambda r: r.total_prompt_tokens,
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
