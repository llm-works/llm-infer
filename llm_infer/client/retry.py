"""Retry helper for transient errors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar

from appinfra import Backoff
from appinfra.log import Logger

from .backends import BackendContext, RetryConfig
from .errors import BackendRequestError, BackendUnavailableError
from .log_utils import fmt_error
from .types import ChatRequest, LLMCallbacks, SendContext, SendResult

# Non-5xx status codes that should trigger retry (5xx are always retried)
# 429 = rate limited, 529 = site overloaded (Cloudflare)
TRANSIENT_4XX_CODES: frozenset[int] = frozenset({429, 529})


class RetryBase:
    """Base class with core retry logic (transient detection, backoff, delay).

    Provides the building blocks for retry behavior without any callback or
    request-type dependencies. Used by RetryHelper (chat clients) and
    EmbeddingClient (embeddings).
    """

    def __init__(self, lg: Logger) -> None:
        self._lg = lg

    def is_transient(self, exc: Exception) -> bool:
        """Check if exception is a transient error that should be retried."""
        if isinstance(exc, BackendUnavailableError):
            return True
        if isinstance(exc, BackendRequestError):
            code = exc.status_code
            if code is None:
                return True
            if 500 <= code < 600:
                return True
            return code in TRANSIENT_4XX_CODES
        return False

    def should_retry(
        self,
        exc: BackendUnavailableError | BackendRequestError,
        start_time: float,
        timeout: float,
    ) -> bool:
        """Check if error should be retried (transient + within timeout)."""
        if not self.is_transient(exc):
            return False
        if timeout > 0 and (time.monotonic() - start_time) >= timeout:
            return False
        return True

    def create_backoff(self, retry: RetryConfig) -> Backoff:
        """Create a fresh Backoff instance from RetryConfig."""
        return Backoff(
            self._lg,
            base=retry.base,
            factor=retry.factor,
            max_delay=retry.max_delay,
        )

    def compute_delay(
        self, backoff: Backoff, timeout: float, start_time: float
    ) -> float | None:
        """Compute delay for next retry, or None if timeout exceeded."""
        delay: float = backoff.next_delay()
        if timeout > 0:
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                return None
            delay = min(delay, remaining)
        return delay


T = TypeVar("T")


def _fire_request(
    lg: Logger,
    callbacks: LLMCallbacks | None,
    request: ChatRequest | None,
    retry: int,
) -> None:
    """Fire on_request callback if configured."""
    if callbacks and callbacks.on_request and request:
        try:
            callbacks.on_request(request, retry)
        except Exception as e:
            lg.warning("on_request callback failed", extra={"exception": e})


def _fire_response(
    lg: Logger,
    callbacks: LLMCallbacks | None,
    request: ChatRequest | None,
    response: Any,
) -> None:
    """Fire on_response callback if configured."""
    if callbacks and callbacks.on_response and request:
        try:
            callbacks.on_response(request, response)
        except Exception as e:
            lg.warning("on_response callback failed", extra={"exception": e})


def _fire_error(
    lg: Logger,
    callbacks: LLMCallbacks | None,
    request: ChatRequest | None,
    error: Exception,
) -> None:
    """Fire on_error callback if configured."""
    if callbacks and callbacks.on_error and request:
        try:
            callbacks.on_error(request, error)
        except Exception as e:
            lg.warning("on_error callback failed", extra={"exception": e})


def _fire_on_before_send(
    lg: Logger,
    callbacks: LLMCallbacks | None,
    ctx: SendContext,
) -> None:
    """Fire on_before_send callback if configured."""
    if callbacks and callbacks.on_before_send:
        try:
            callbacks.on_before_send(ctx)
        except Exception as e:
            lg.warning("on_before_send callback failed", extra={"exception": e})


def _fire_on_after_send(
    lg: Logger,
    callbacks: LLMCallbacks | None,
    ctx: SendContext,
    result: SendResult,
) -> None:
    """Fire on_after_send callback if configured."""
    if callbacks and callbacks.on_after_send:
        try:
            callbacks.on_after_send(ctx, result)
        except Exception as e:
            lg.warning("on_after_send callback failed", extra={"exception": e})


def _retry_reason(error: BackendUnavailableError | BackendRequestError) -> str:
    """Determine retry reason from error type."""
    if isinstance(error, BackendUnavailableError):
        return "unavailable"
    if isinstance(error, BackendRequestError):
        code = error.status_code
        if code == 429:
            return "rate_limit"
        if code == 529:
            return "overloaded"
        if code is not None and 500 <= code < 600:
            return "server_error"
    return "timeout"


class RetryHelper(RetryBase):
    """Handles retry logic with exponential backoff and callback support.

    Extends RetryBase with callback firing for chat client observability.
    Creates a fresh Backoff instance per-request from RetryConfig to avoid
    race conditions when multiple clients share a backend.
    """

    def __init__(self, lg: Logger, ctx: BackendContext, provider: str) -> None:
        super().__init__(lg)
        self._ctx = ctx
        self._provider = provider

    def _log_retry(
        self,
        error: BackendUnavailableError | BackendRequestError,
        attempt: int,
        delay: float,
        req_id: str | None = None,
        model: str | None = None,
    ) -> None:
        """Log retry attempt with context."""
        status_code = None
        if isinstance(error, BackendRequestError):
            status_code = error.status_code

        extra: dict[str, Any] = {
            "provider": self._provider,
            "model": model,
            "error_type": type(error).__name__,
            "status_code": status_code,
            "error": fmt_error(error),
            "retry_attempt": attempt,
            "delay_seconds": round(delay, 2),
        }
        if req_id:
            extra["req"] = req_id

        self._lg.warning("transient error, retrying", extra=extra)

    def _log_retry_send(
        self,
        attempt: int,
        model: str | None,
        req_id: str | None = None,
    ) -> None:
        """Log debug message before retry send."""
        extra: dict[str, Any] = {"attempt": attempt}
        if model:
            extra["model"] = model
        if req_id:
            extra["req"] = req_id

        self._lg.debug("retrying request...", extra=extra)

    def _send_context(
        self,
        attempt: int,
        model: str | None,
        req_id: str | None,
        last_error: BackendUnavailableError | BackendRequestError | None = None,
        last_delay: float | None = None,
    ) -> SendContext:
        """Build SendContext for on_before_send/on_after_send callbacks."""
        return SendContext(
            attempt=attempt,
            retry_reason=_retry_reason(last_error) if last_error else None,
            delay_seconds=last_delay,
            model=model,
            backend=self._provider,
            req_id=req_id,
        )

    def _on_after_send(
        self,
        callbacks: LLMCallbacks | None,
        ctx: SendContext,
        start: float,
        error: Exception | None = None,
    ) -> None:
        """Fire on_after_send callback with timing."""
        elapsed_ms = (time.monotonic() - start) * 1000
        status_code = 200 if error is None else None
        if isinstance(error, BackendRequestError):
            status_code = error.status_code
        _fire_on_after_send(
            self._lg, callbacks, ctx, SendResult(status_code, error, elapsed_ms)
        )

    def _retry_delay(
        self,
        e: BackendUnavailableError | BackendRequestError,
        backoff: Backoff,
        retry: RetryConfig,
        start_time: float,
    ) -> float | None:
        """Compute delay if retry is possible, else None."""
        if not self.should_retry(e, start_time, retry.timeout):
            return None
        return self.compute_delay(backoff, retry.timeout, start_time)

    def _prep_retry(
        self,
        e: BackendUnavailableError | BackendRequestError,
        delay: float,
        retry_count: int,
        req_id: str | None,
        model: str | None,
    ) -> None:
        """Log and sleep for retry. Call after incrementing retry_count."""
        self._log_retry(e, retry_count, delay, req_id, model)
        time.sleep(delay)
        self._log_retry_send(retry_count, model, req_id)

    async def _prep_retry_async(
        self,
        e: BackendUnavailableError | BackendRequestError,
        delay: float,
        retry_count: int,
        req_id: str | None,
        model: str | None,
    ) -> None:
        """Log and sleep for retry (async)."""
        self._log_retry(e, retry_count, delay, req_id, model)
        await asyncio.sleep(delay)
        self._log_retry_send(retry_count, model, req_id)

    def call(
        self,
        fn: Callable[[], T],
        request: ChatRequest | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> T:
        """Execute fn with retry on transient errors."""
        retry_count = 0
        _fire_request(self._lg, callbacks, request, retry_count)

        retry = self._ctx.retry
        if retry is None:
            return self._call_no_retry(fn, request, callbacks)

        return self._call_with_retry(fn, request, callbacks, retry, retry_count)

    def _call_no_retry(
        self,
        fn: Callable[[], T],
        request: ChatRequest | None,
        callbacks: LLMCallbacks | None,
    ) -> T:
        """Execute without retry, firing callbacks."""
        req_id = request.id if request else None
        model = request.model if request else None
        ctx = self._send_context(1, model, req_id)
        _fire_on_before_send(self._lg, callbacks, ctx)
        start = time.monotonic()
        try:
            result = fn()
            self._on_after_send(callbacks, ctx, start)
            _fire_response(self._lg, callbacks, request, result)
            return result
        except Exception as e:
            self._on_after_send(callbacks, ctx, start, e)
            _fire_error(self._lg, callbacks, request, e)
            raise

    def _call_with_retry(
        self,
        fn: Callable[[], T],
        request: ChatRequest | None,
        callbacks: LLMCallbacks | None,
        retry: RetryConfig,
        retry_count: int,
    ) -> T:
        """Execute with retry loop, firing callbacks."""
        backoff, t0 = self.create_backoff(retry), time.monotonic()
        req_id = request.id if request else None
        model = request.model if request else None
        last_err: BackendUnavailableError | BackendRequestError | None = None
        last_dly: float | None = None
        while True:
            ctx = self._send_context(retry_count + 1, model, req_id, last_err, last_dly)
            _fire_on_before_send(self._lg, callbacks, ctx)
            start = time.monotonic()
            try:
                result = fn()
                self._on_after_send(callbacks, ctx, start)
                _fire_response(self._lg, callbacks, request, result)
                return result
            except (BackendUnavailableError, BackendRequestError) as e:
                self._on_after_send(callbacks, ctx, start, e)
                dly = self._retry_delay(e, backoff, retry, t0)
                if dly is None:
                    _fire_error(self._lg, callbacks, request, e)
                    raise
                retry_count += 1
                last_err, last_dly = e, dly
                self._prep_retry(e, dly, retry_count, req_id, model)
                _fire_request(self._lg, callbacks, request, retry_count)
            except Exception as e:
                self._on_after_send(callbacks, ctx, start, e)
                _fire_error(self._lg, callbacks, request, e)
                raise

    async def call_async(
        self,
        fn: Callable[[], Awaitable[T]],
        request: ChatRequest | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> T:
        """Execute async fn with retry on transient errors."""
        retry_count = 0
        _fire_request(self._lg, callbacks, request, retry_count)

        retry = self._ctx.retry
        if retry is None:
            return await self._call_no_retry_async(fn, request, callbacks)

        return await self._call_with_retry_async(
            fn, request, callbacks, retry, retry_count
        )

    async def _call_no_retry_async(
        self,
        fn: Callable[[], Awaitable[T]],
        request: ChatRequest | None,
        callbacks: LLMCallbacks | None,
    ) -> T:
        """Execute without retry, firing callbacks (async)."""
        req_id = request.id if request else None
        model = request.model if request else None
        ctx = self._send_context(1, model, req_id)
        _fire_on_before_send(self._lg, callbacks, ctx)
        start = time.monotonic()
        try:
            result = await fn()
            self._on_after_send(callbacks, ctx, start)
            _fire_response(self._lg, callbacks, request, result)
            return result
        except Exception as e:
            self._on_after_send(callbacks, ctx, start, e)
            _fire_error(self._lg, callbacks, request, e)
            raise

    async def _call_with_retry_async(
        self,
        fn: Callable[[], Awaitable[T]],
        request: ChatRequest | None,
        callbacks: LLMCallbacks | None,
        retry: RetryConfig,
        retry_count: int,
    ) -> T:
        """Execute with retry loop, firing callbacks (async)."""
        backoff, t0 = self.create_backoff(retry), time.monotonic()
        req_id = request.id if request else None
        model = request.model if request else None
        last_err: BackendUnavailableError | BackendRequestError | None = None
        last_dly: float | None = None
        while True:
            ctx = self._send_context(retry_count + 1, model, req_id, last_err, last_dly)
            _fire_on_before_send(self._lg, callbacks, ctx)
            start = time.monotonic()
            try:
                result = await fn()
                self._on_after_send(callbacks, ctx, start)
                _fire_response(self._lg, callbacks, request, result)
                return result
            except (BackendUnavailableError, BackendRequestError) as e:
                self._on_after_send(callbacks, ctx, start, e)
                dly = self._retry_delay(e, backoff, retry, t0)
                if dly is None:
                    _fire_error(self._lg, callbacks, request, e)
                    raise
                retry_count += 1
                last_err, last_dly = e, dly
                await self._prep_retry_async(e, dly, retry_count, req_id, model)
                _fire_request(self._lg, callbacks, request, retry_count)
            except Exception as e:
                self._on_after_send(callbacks, ctx, start, e)
                _fire_error(self._lg, callbacks, request, e)
                raise

    def _stream_retry_delay(
        self,
        e: BackendUnavailableError | BackendRequestError,
        backoff: Backoff,
        retry: RetryConfig,
        start_time: float,
    ) -> float | None:
        """Compute delay before retrying a pre-first-token stream error.

        Returns None when the error is not transient or the retry budget is
        exhausted, in which case the caller must re-raise.
        """
        return self._retry_delay(e, backoff, retry, start_time)

    def stream(
        self,
        fn: Callable[[], Iterator[str]],
        request: ChatRequest | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> Iterator[str]:
        """Execute streaming fn with retry until the first token is yielded.

        Retries the same transient errors as call(), but only while no token
        has been yielded yet. Once streaming has started, errors propagate
        unchanged: partial output cannot be replayed safely. fn is invoked
        fresh on every attempt.

        Unlike call(), the caller is responsible for firing the initial
        on_request callback (retry_count=0); this method fires on_request
        only on retries. This avoids duplicate callbacks when the client
        layer manages the initial event.

        The backoff sleep is a blocking time.sleep and cannot be interrupted;
        abort/cancellation during backoff is only supported on the async path
        (stream_async).
        """
        retry = self._ctx.retry
        req_id = request.id if request else None
        model = request.model if request else None

        if retry is None:
            yield from self._stream_no_retry(fn, model, req_id, callbacks)
            return

        yield from self._stream_with_retry(fn, request, model, req_id, callbacks, retry)

    def _stream_no_retry(
        self,
        fn: Callable[[], Iterator[str]],
        model: str | None,
        req_id: str | None,
        callbacks: LLMCallbacks | None,
    ) -> Iterator[str]:
        """Stream without retry, firing send callbacks."""
        ctx = self._send_context(1, model, req_id)
        _fire_on_before_send(self._lg, callbacks, ctx)
        start = time.monotonic()
        try:
            it = fn()
            first = next(it)
        except StopIteration:
            self._on_after_send(callbacks, ctx, start)
            return
        except Exception as e:
            self._on_after_send(callbacks, ctx, start, e)
            raise
        self._on_after_send(callbacks, ctx, start)
        yield first
        yield from it

    def _stream_with_retry(
        self,
        fn: Callable[[], Iterator[str]],
        request: ChatRequest | None,
        model: str | None,
        req_id: str | None,
        callbacks: LLMCallbacks | None,
        retry: RetryConfig,
    ) -> Iterator[str]:
        """Stream with retry loop, firing send callbacks."""
        backoff, t0, retry_count = self.create_backoff(retry), time.monotonic(), 0
        last_err: BackendUnavailableError | BackendRequestError | None = None
        last_dly: float | None = None
        while True:
            ctx = self._send_context(retry_count + 1, model, req_id, last_err, last_dly)
            _fire_on_before_send(self._lg, callbacks, ctx)
            start = time.monotonic()
            try:
                it = fn()
                first = next(it)
            except StopIteration:
                self._on_after_send(callbacks, ctx, start)
                return
            except (BackendUnavailableError, BackendRequestError) as e:
                self._on_after_send(callbacks, ctx, start, e)
                dly = self._stream_retry_delay(e, backoff, retry, t0)
                if dly is None:
                    raise
                last_err, last_dly, retry_count = e, dly, retry_count + 1
                self._prep_retry(e, dly, retry_count, req_id, model)
                _fire_request(self._lg, callbacks, request, retry_count)
                continue
            except Exception as e:
                self._on_after_send(callbacks, ctx, start, e)
                raise
            self._on_after_send(callbacks, ctx, start)
            yield first
            yield from it
            return

    async def stream_async(
        self,
        fn: Callable[[], AsyncIterator[str]],
        request: ChatRequest | None = None,
        callbacks: LLMCallbacks | None = None,
    ) -> AsyncIterator[str]:
        """Async variant of stream(): retry until the first token is yielded.

        Like stream(), the caller fires the initial on_request callback.

        The backoff sleep uses asyncio.sleep, so task cancellation (e.g. an
        abort signal racing the stream task) propagates promptly.
        """
        retry = self._ctx.retry
        req_id = request.id if request else None
        model = request.model if request else None

        if retry is None:
            async for token in self._stream_no_retry_async(
                fn, model, req_id, callbacks
            ):
                yield token
            return

        async for token in self._stream_with_retry_async(
            fn, request, model, req_id, callbacks, retry
        ):
            yield token

    async def _stream_no_retry_async(
        self,
        fn: Callable[[], AsyncIterator[str]],
        model: str | None,
        req_id: str | None,
        callbacks: LLMCallbacks | None,
    ) -> AsyncIterator[str]:
        """Async stream without retry, firing send callbacks."""
        ctx = self._send_context(1, model, req_id)
        _fire_on_before_send(self._lg, callbacks, ctx)
        start = time.monotonic()
        try:
            it = fn()
            first = await anext(it)
        except StopAsyncIteration:
            self._on_after_send(callbacks, ctx, start)
            return
        except Exception as e:
            self._on_after_send(callbacks, ctx, start, e)
            raise
        self._on_after_send(callbacks, ctx, start)
        yield first
        async for token in it:
            yield token

    async def _stream_with_retry_async(
        self,
        fn: Callable[[], AsyncIterator[str]],
        request: ChatRequest | None,
        model: str | None,
        req_id: str | None,
        callbacks: LLMCallbacks | None,
        retry: RetryConfig,
    ) -> AsyncIterator[str]:
        """Async stream with retry loop, firing send callbacks."""
        backoff, t0, retry_count = self.create_backoff(retry), time.monotonic(), 0
        last_err: BackendUnavailableError | BackendRequestError | None = None
        last_dly: float | None = None
        while True:
            ctx = self._send_context(retry_count + 1, model, req_id, last_err, last_dly)
            _fire_on_before_send(self._lg, callbacks, ctx)
            start = time.monotonic()
            try:
                it = fn()
                first = await anext(it)
            except StopAsyncIteration:
                self._on_after_send(callbacks, ctx, start)
                return
            except (BackendUnavailableError, BackendRequestError) as e:
                self._on_after_send(callbacks, ctx, start, e)
                dly = self._stream_retry_delay(e, backoff, retry, t0)
                if dly is None:
                    raise
                last_err, last_dly, retry_count = e, dly, retry_count + 1
                await self._prep_retry_async(e, dly, retry_count, req_id, model)
                _fire_request(self._lg, callbacks, request, retry_count)
                continue
            except Exception as e:
                self._on_after_send(callbacks, ctx, start, e)
                raise
            self._on_after_send(callbacks, ctx, start)
            yield first
            async for token in it:
                yield token
            return
