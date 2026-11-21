# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""AsyncRequestTrackingMixin: in-flight request counting and drain on close."""

from __future__ import annotations

import asyncio

import pytest

from llm_infer.client.backends.mixins import AsyncRequestTrackingMixin
from llm_infer.client.errors import BackendUnavailableError


class _Tracked(AsyncRequestTrackingMixin):
    def __init__(self) -> None:
        self._init_async_tracking()


def test_drain_without_requests_returns_at_once() -> None:
    tracked = _Tracked()
    asyncio.run(tracked._drain_async_requests())
    with pytest.raises(BackendUnavailableError):
        tracked._acquire_async_request()


def test_every_waiting_close_is_woken() -> None:
    tracked = _Tracked()

    async def run() -> None:
        tracked._acquire_async_request()
        closes = [
            asyncio.create_task(tracked._drain_async_requests()) for _ in range(2)
        ]
        await asyncio.sleep(0)  # both closes reach their wait
        assert len(tracked._drain_waiters) == 2
        tracked._release_async_request()
        await asyncio.gather(*closes)

    asyncio.run(run())
    assert tracked._drain_waiters == []


def test_release_skips_waiter_whose_loop_closed() -> None:
    tracked = _Tracked()
    tracked._acquire_async_request()
    loop = asyncio.new_event_loop()
    loop.close()
    tracked._drain_waiters.append((loop, asyncio.Event()))
    tracked._release_async_request()  # must not raise from the request's finally
    assert tracked._drain_waiters == []
