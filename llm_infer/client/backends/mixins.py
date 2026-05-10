"""Reusable mixins for backend implementations."""

from __future__ import annotations

import asyncio

from ..errors import BackendUnavailableError


class AsyncRequestTrackingMixin:
    """Mixin for tracking in-flight async requests during close.

    Backends with lazy async clients should use this mixin to prevent
    closing the client while requests are still in-flight. The mixin
    provides reference counting for active requests and graceful drain
    on close.

    Note: This mixin assumes single-threaded asyncio. The counter and
    close flag are not thread-safe. Do not share backend instances
    across multiple event loops or OS threads.

    Usage:
        class MyBackend(AsyncRequestTrackingMixin, Backend):
            def __init__(self, ...):
                super().__init__(...)
                self._init_async_tracking()

            async def _execute_async(self, ...):
                self._acquire_async_request()  # raises if closing
                try:
                    client = self._get_async_client()
                    # ... do request
                finally:
                    self._release_async_request()

            async def aclose(self):
                await self._drain_async_requests()
                # ... close clients
    """

    _active_async_requests: int
    _close_requested: bool
    _drain_event: asyncio.Event | None

    def _init_async_tracking(self) -> None:
        """Initialize async request tracking state. Call from __init__."""
        self._active_async_requests = 0
        self._close_requested = False
        self._drain_event = None

    def _acquire_async_request(self) -> None:
        """Acquire a request slot, raising if backend is closing.

        This atomically checks the close flag and increments the counter,
        preventing races between new requests and close.

        Raises:
            BackendUnavailableError: If close has been requested.
        """
        if self._close_requested:
            raise BackendUnavailableError(
                "Backend is closing, cannot accept new requests"
            )
        self._active_async_requests += 1

    def _release_async_request(self) -> None:
        """Track completion of an async request. Call in finally block."""
        self._active_async_requests -= 1
        if self._active_async_requests == 0 and self._drain_event is not None:
            self._drain_event.set()

    async def _drain_async_requests(self) -> None:
        """Wait for all in-flight async requests to complete. Call from aclose()."""
        self._close_requested = True
        if self._active_async_requests > 0:
            if self._drain_event is None:
                self._drain_event = asyncio.Event()
            await self._drain_event.wait()
