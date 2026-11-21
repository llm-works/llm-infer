# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""LoopClients keeps one async client per event loop."""

from __future__ import annotations

import asyncio
import contextlib
import gc
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client.backends.loop_clients import LoopClients


class _Client:
    """Stand-in for an async HTTP client."""


class _LoopBoundClient(_Client):
    """Holds its loop, as an httpx client does through its pooled connections."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()


@pytest.fixture
def closer() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def holder(lg: Logger, closer: AsyncMock) -> LoopClients[_Client]:
    return LoopClients(lg, _Client, closer)


@pytest.fixture
def other_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _get_on(loop: asyncio.AbstractEventLoop, holder: LoopClients[_Client]) -> _Client:
    async def get() -> _Client:
        return holder.get()

    return loop.run_until_complete(get())


class TestGet:
    def test_same_loop_gets_same_client(self, holder: LoopClients[_Client]) -> None:
        async def twice() -> tuple[_Client, _Client, int]:
            return holder.get(), holder.get(), len(holder)

        first, second, held = asyncio.run(twice())
        assert first is second
        assert held == 1

    def test_each_loop_gets_its_own_client(
        self, holder: LoopClients[_Client], other_loop: asyncio.AbstractEventLoop
    ) -> None:
        loop = asyncio.new_event_loop()
        try:
            first = _get_on(loop, holder)
            second = _get_on(other_loop, holder)
        finally:
            loop.close()
        assert first is not second

    def test_outside_a_loop_raises(self, holder: LoopClients[_Client]) -> None:
        with pytest.raises(RuntimeError, match="no running event loop"):
            holder.get()

    def test_client_dropped_with_its_loop(self, holder: LoopClients[_Client]) -> None:
        loop = asyncio.new_event_loop()
        _get_on(loop, holder)
        loop.close()
        del loop
        gc.collect()
        assert len(holder) == 0

    def test_client_of_closed_loop_evicted_on_next_get(
        self, lg: Logger, closer: AsyncMock
    ) -> None:
        # The client keeps its loop alive, so only eviction by is_closed() frees it.
        holder: LoopClients[_Client] = LoopClients(lg, _LoopBoundClient, closer)
        loop = asyncio.new_event_loop()
        _get_on(loop, holder)
        loop.close()
        del loop
        gc.collect()

        async def get_then_count() -> int:
            holder.get()
            return len(holder)

        assert asyncio.run(get_then_count()) == 1
        closer.assert_not_awaited()


class TestAclose:
    def test_closes_running_loops_client(
        self, holder: LoopClients[_Client], closer: AsyncMock
    ) -> None:
        async def use_then_close() -> _Client:
            client = holder.get()
            await holder.aclose()
            return client

        client = asyncio.run(use_then_close())
        closer.assert_awaited_once_with(client)
        assert len(holder) == 0

    def test_drops_client_of_closed_loop(
        self,
        holder: LoopClients[_Client],
        closer: AsyncMock,
        other_loop: asyncio.AbstractEventLoop,
    ) -> None:
        _get_on(other_loop, holder)
        other_loop.close()
        asyncio.run(holder.aclose())
        closer.assert_not_awaited()
        assert len(holder) == 0

    def test_loop_closed_after_running_check(
        self, holder: LoopClients[_Client], closer: AsyncMock
    ) -> None:
        # The other loop reports running, then refuses the close callback.
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        loop.is_running.return_value = True
        loop.call_soon_threadsafe.side_effect = RuntimeError("Event loop is closed")
        holder._clients[loop] = _Client()
        asyncio.run(holder.aclose())
        loop.call_soon_threadsafe.assert_called_once()
        closer.assert_not_awaited()
        assert len(holder) == 0
        assert holder._pending == {}

    def test_cancelled_aclose_leaves_scheduled_close_running(self, lg: Logger) -> None:
        async def run() -> list[_Client]:
            release = asyncio.Event()
            closed: list[_Client] = []

            async def close(client: _Client) -> None:
                await release.wait()
                closed.append(client)

            holder = LoopClients(lg, _Client, close)
            client = _Client()
            with holder._lock:  # as an aclose() on another loop schedules it
                holder._schedule_close(asyncio.get_running_loop(), client)
            waiter = asyncio.create_task(holder.aclose())
            await asyncio.sleep(0)  # the waiter reaches its wait on the close
            assert not waiter.done()
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter
            release.set()
            await holder.aclose()  # still finds the close and waits for it
            return closed

        closed = asyncio.run(run())
        assert len(closed) == 1

    def test_failed_close_on_other_loop_is_logged(self) -> None:
        error = OSError("reset")
        lg = MagicMock(spec=Logger)
        holder = LoopClients(lg, _Client, AsyncMock(side_effect=error))
        asyncio.run(holder._close_logged(_Client()))
        lg.warning.assert_called_once_with(
            "async client close failed", extra={"exception": error}
        )
