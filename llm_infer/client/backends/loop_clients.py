# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Async clients kept per event loop."""

from __future__ import annotations

import asyncio
import contextlib
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

    Clients are held by weak reference to their loop, so a loop that is garbage
    collected takes its client with it.
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
        self._lock = threading.Lock()
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = (
            weakref.WeakKeyDictionary()
        )
        self._closing: set[asyncio.Task[None]] = set()

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
                client = self._factory()
                self._clients[loop] = client
            return client

    async def aclose(self) -> None:
        """Close every client, each on its own loop.

        The running loop's client is closed before this returns. A client of
        another loop that is still running is closed by a task scheduled on that
        loop, without waiting for it: that loop may stop at any time, and waiting
        on it could then never end. A client whose loop is stopped or closed
        cannot run its close; it is dropped, and garbage collection releases its
        sockets.
        """
        with self._lock:
            clients = list(self._clients.items())
            self._clients.clear()
        current = asyncio.get_running_loop()
        for loop, client in clients:
            if loop is current:
                await self._close(client)
            elif loop.is_running():
                self._schedule_close(loop, client)

    def _schedule_close(self, loop: asyncio.AbstractEventLoop, client: T) -> None:
        """Close the client by a task on its own loop."""

        def start() -> None:
            task = loop.create_task(self._close_logged(client))
            self._closing.add(task)  # create_task keeps only a weak reference
            task.add_done_callback(self._closing.discard)

        # A loop that closed after the is_running() check raises; its client is
        # dropped like one whose loop was already closed.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(start)

    async def _close_logged(self, client: T) -> None:
        """Close the client; a failure is logged, as no caller awaits the task."""
        try:
            await self._close(client)
        except Exception as e:
            self._lg.warning("async client close failed", extra={"exception": e})


def httpx_clients(lg: Logger, **client_kwargs: Any) -> LoopClients[httpx.AsyncClient]:
    """LoopClients of ``httpx.AsyncClient(**client_kwargs)``."""
    return LoopClients(
        lg, lambda: httpx.AsyncClient(**client_kwargs), httpx.AsyncClient.aclose
    )
