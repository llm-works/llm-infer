"""Retry helper for transient errors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from appinfra import Backoff
from appinfra.log import Logger

from .backends import BackendContext, RetryConfig
from .errors import BackendRequestError, BackendUnavailableError
from .types import ChatRequest, LLMCallbacks

# Non-5xx status codes that should trigger retry (5xx are always retried)
# 429 = rate limited, 529 = site overloaded (Cloudflare)
TRANSIENT_4XX_CODES: frozenset[int] = frozenset({429, 529})

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


class RetryHelper:
    """Handles retry logic with exponential backoff.

    Creates a fresh Backoff instance per-request from RetryConfig to avoid
    race conditions when multiple clients share a backend.
    """

    def __init__(self, lg: Logger, ctx: BackendContext) -> None:
        self._lg = lg
        self._ctx = ctx

    def _is_transient(self, exc: Exception) -> bool:
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

    def _should_retry(
        self,
        e: BackendUnavailableError | BackendRequestError,
        start_time: float,
        timeout: float,
    ) -> bool:
        """Check if error should be retried."""
        if not self._is_transient(e):
            return False
        if timeout > 0 and (time.monotonic() - start_time) >= timeout:
            return False
        return True

    def _create_backoff(self, retry: RetryConfig) -> Backoff:
        """Create a fresh Backoff instance from RetryConfig."""
        return Backoff(
            self._lg,
            base=retry.base,
            factor=retry.factor,
            max_delay=retry.max_delay,
        )

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
        try:
            result = fn()
            _fire_response(self._lg, callbacks, request, result)
            return result
        except Exception as e:
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
        backoff = self._create_backoff(retry)
        start_time = time.monotonic()
        while True:
            try:
                result = fn()
                _fire_response(self._lg, callbacks, request, result)
                return result
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._should_retry(e, start_time, retry.timeout):
                    _fire_error(self._lg, callbacks, request, e)
                    raise
                delay = self._compute_delay(backoff, retry.timeout, start_time)
                if delay is None:
                    _fire_error(self._lg, callbacks, request, e)
                    raise
                time.sleep(delay)
                retry_count += 1
                _fire_request(self._lg, callbacks, request, retry_count)
            except Exception as e:
                _fire_error(self._lg, callbacks, request, e)
                raise

    def _compute_delay(
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
        try:
            result = await fn()
            _fire_response(self._lg, callbacks, request, result)
            return result
        except Exception as e:
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
        backoff = self._create_backoff(retry)
        start_time = time.monotonic()
        while True:
            try:
                result = await fn()
                _fire_response(self._lg, callbacks, request, result)
                return result
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._should_retry(e, start_time, retry.timeout):
                    _fire_error(self._lg, callbacks, request, e)
                    raise
                delay = self._compute_delay(backoff, retry.timeout, start_time)
                if delay is None:
                    _fire_error(self._lg, callbacks, request, e)
                    raise
                await asyncio.sleep(delay)
                retry_count += 1
                _fire_request(self._lg, callbacks, request, retry_count)
            except Exception as e:
                _fire_error(self._lg, callbacks, request, e)
                raise
