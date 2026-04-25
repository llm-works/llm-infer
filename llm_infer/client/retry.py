"""Retry helper for transient errors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from appinfra import Backoff
from appinfra.log import Logger

from .backends import BackendContext, RetryConfig
from .errors import BackendRequestError, BackendUnavailableError

# Non-5xx status codes that should trigger retry (5xx are always retried)
# 429 = rate limited, 529 = site overloaded (Cloudflare)
TRANSIENT_4XX_CODES: frozenset[int] = frozenset({429, 529})

T = TypeVar("T")


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

    def call(self, fn: Callable[[], T]) -> T:
        """Execute fn with retry on transient errors."""
        retry = self._ctx.retry
        if retry is None:
            return fn()
        backoff = self._create_backoff(retry)
        start_time = time.monotonic()
        while True:
            try:
                return fn()
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._should_retry(e, start_time, retry.timeout):
                    raise
                delay = backoff.next_delay()
                if retry.timeout > 0:
                    remaining = retry.timeout - (time.monotonic() - start_time)
                    if remaining <= 0:
                        raise
                    delay = min(delay, remaining)
                time.sleep(delay)

    async def call_async(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Execute async fn with retry on transient errors."""
        retry = self._ctx.retry
        if retry is None:
            return await fn()
        backoff = self._create_backoff(retry)
        start_time = time.monotonic()
        while True:
            try:
                return await fn()
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._should_retry(e, start_time, retry.timeout):
                    raise
                delay = backoff.next_delay()
                if retry.timeout > 0:
                    remaining = retry.timeout - (time.monotonic() - start_time)
                    if remaining <= 0:
                        raise
                    delay = min(delay, remaining)
                await asyncio.sleep(delay)
