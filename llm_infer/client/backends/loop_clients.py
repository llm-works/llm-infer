# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Async clients kept per event loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import threading
import weakref
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar

import httpx
from appinfra.log import Logger

T = TypeVar("T")


class LoopClients(Generic[T]):
    """One async client per event loop, created on that loop's first use.

    An async HTTP client is bound to the loop that first used it: its pooled
    connections hold that loop's transports, and another loop using them fails
    (``RuntimeError: Event loop is closed`` once the first loop is gone). Keeping
    one client per loop lets one backend serve threads that each run their own
    loop, with connection reuse within each loop.

    A client usually keeps its loop alive (its pooled connections hold the
    loop's transports), so clients of closed loops are dropped explicitly,
    whenever a client is created for a new loop.
    """

    def __init__(
        self,
        lg: Logger,
        factory: Callable[[], T],
        close: Callable[[T], Coroutine[Any, Any, None]],
    ) -> None:
        """Initialize the holder.

        Args:
            lg: Logger instance.
            factory: Builds a client; called on the loop that will use it.
            close: Closes a client; awaited on the client's own loop.
        """
        self._lg = lg
        self._factory = factory
        self._close = close
        # Reentrant: a close that already finished runs _forget() at once, from
        # add_done_callback() in _schedule_close(), with the lock held.
        self._lock = threading.RLock()
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = (
            weakref.WeakKeyDictionary()
        )
        # Closes scheduled on a loop by an aclose() on another loop
        self._pending: dict[
            asyncio.AbstractEventLoop, set[concurrent.futures.Future[None]]
        ] = {}

    def __len__(self) -> int:
        """Number of loops holding a client."""
        with self._lock:
            return len(self._clients)

    def get(self) -> T:
        """The running loop's client, created on its first use.

        Raises:
            RuntimeError: If no event loop is running in this thread.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            client = self._clients.get(loop)
            if client is None:
                self._evict_closed()
                client = self._factory()
                self._clients[loop] = client
            return client

    def _evict_closed(self) -> None:
        """Drop the clients of closed loops. Call with the lock held.

        A closed loop cannot run its client's close; dropping the client lets
        garbage collection release its sockets.
        """
        for loop in [loop for loop in self._clients if loop.is_closed()]:
            del self._clients[loop]
        # A close scheduled on a loop that closed first never runs
        for loop in [loop for loop in self._pending if loop.is_closed()]:
            del self._pending[loop]

    async def aclose(self) -> None:
        """Close every client, each on its own loop.

        The running loop's client is closed before this returns, also when an
        aclose() on another loop took it first. A client of another loop that is
        still running is closed by a task scheduled on that loop, without waiting
        for it: that loop may stop at any time, and waiting on it could then
        never end. A client whose loop is stopped or closed cannot run its close;
        it is dropped, and garbage collection releases its sockets.
        """
        current = asyncio.get_running_loop()
        own: list[T] = []
        with self._lock:
            for loop, client in list(self._clients.items()):
                if loop is current:
                    own.append(client)
                elif loop.is_running():
                    self._schedule_close(loop, client)
            self._clients.clear()
            # Left recorded until each close finishes (_forget), for a later aclose()
            scheduled_here = set(self._pending.get(current, ()))
        for client in own:
            await self._close(client)
        if scheduled_here:
            # Shielded: cancelling this wait must not cancel the closes themselves.
            await asyncio.shield(
                asyncio.gather(*(asyncio.wrap_future(f) for f in scheduled_here))
            )

    def _schedule_close(self, loop: asyncio.AbstractEventLoop, client: T) -> None:
        """Close the client on its own loop. Call with the lock held.

        The future is recorded so an aclose() on that loop can wait for it; it
        also keeps the close task alive, which its loop references only weakly.
        """
        close = self._close_logged(client)
        try:
            future = asyncio.run_coroutine_threadsafe(close, loop)
        except RuntimeError:
            # The loop closed after the is_running() check: its client is
            # dropped like one whose loop was already closed.
            close.close()
            return
        self._pending.setdefault(loop, set()).add(future)
        future.add_done_callback(functools.partial(self._forget, loop))

    def _forget(
        self, loop: asyncio.AbstractEventLoop, future: concurrent.futures.Future[None]
    ) -> None:
        """Drop a finished close from the pending ones of its loop."""
        with self._lock:
            pending = self._pending.get(loop)
            if pending is not None:
                pending.discard(future)
                if not pending:
                    del self._pending[loop]

    async def _close_logged(self, client: T) -> None:
        """Close the client; a failure is logged, as the aclose() that scheduled
        the close does not wait for it."""
        try:
            await self._close(client)
        except Exception as e:
            self._lg.warning("async client close failed", extra={"exception": e})


def httpx_clients(lg: Logger, **client_kwargs: Any) -> LoopClients[httpx.AsyncClient]:
    """LoopClients of ``httpx.AsyncClient(**client_kwargs)``."""
    return LoopClients(
        lg, lambda: httpx.AsyncClient(**client_kwargs), httpx.AsyncClient.aclose
    )
