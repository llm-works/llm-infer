# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Cancelling engine-side work when the API side gives up on a request."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger
from appinfra.service.errors import ChannelError, ChannelTimeoutError
from fastapi.responses import JSONResponse

from llm_infer.serving.api import cancel as cancel_module
from llm_infer.serving.api.errors import submit_inference
from llm_infer.serving.api.openai.streaming_generators import ChatStreamingGenerator
from llm_infer.serving.dispatch.types import (
    CancelRequest,
    CancelResponse,
    Request,
    RequestStatus,
    Response,
    StreamChunk,
)


class _IPC:
    """IPC stub: inference behaves as `on_submit` says; cancels are recorded."""

    def __init__(self, on_submit: Callable[[], Awaitable[Any]]) -> None:
        self._on_submit = on_submit
        self.cancelled: list[str] = []

    async def submit(self, request: Any, timeout: float | None = None) -> Any:
        if isinstance(request, CancelRequest):
            self.cancelled.append(request.request_id)
            return CancelResponse(id=request.id, found=True)
        return await self._on_submit()

    async def submit_stream(self, request: Any) -> Any:
        result = await self._on_submit()
        yield result


class _HTTPRequest:
    """Starlette request stand-in: disconnects after `disconnect_after` s."""

    def __init__(self, disconnect_after: float | None) -> None:
        self._after = disconnect_after

    async def receive(self) -> dict[str, Any]:
        if self._after is None:
            await asyncio.Event().wait()  # never disconnects
        await asyncio.sleep(self._after or 0)
        return {"type": "http.disconnect"}


async def _forever() -> Any:
    await asyncio.Event().wait()


async def _drain_cancels() -> None:
    """Let send_cancel's background task run."""
    await asyncio.gather(*list(cancel_module._pending))


def _request() -> Request:
    return Request(id="r1", prompt="hi", max_tokens=10)


async def _submit(ipc: _IPC, disconnect_after: float | None) -> Any:
    http_request: Any = _HTTPRequest(disconnect_after)
    return await asyncio.wait_for(
        submit_inference(MagicMock(spec=Logger), ipc, _request(), http_request),
        timeout=5,
    )


@pytest.mark.asyncio
async def test_success_returns_response_without_cancel() -> None:
    done = Response(id="r1", status=RequestStatus.COMPLETED, result="ok")

    async def answer() -> Response:
        return done

    ipc = _IPC(answer)
    result = await _submit(ipc, None)

    await _drain_cancels()
    assert result is done
    assert ipc.cancelled == []


@pytest.mark.asyncio
async def test_timeout_cancels_and_returns_504() -> None:
    async def time_out() -> Any:
        raise ChannelTimeoutError("Request r1 timed out after 300s")

    ipc = _IPC(time_out)
    result = await _submit(ipc, None)

    await _drain_cancels()
    assert isinstance(result, JSONResponse)
    assert result.status_code == 504
    assert ipc.cancelled == ["r1"]


@pytest.mark.asyncio
async def test_client_disconnect_cancels() -> None:
    ipc = _IPC(_forever)
    result = await _submit(ipc, 0.05)

    await _drain_cancels()
    assert isinstance(result, JSONResponse)
    assert json.loads(bytes(result.body))["error"]["code"] == "client_closed"
    assert ipc.cancelled == ["r1"]


@pytest.mark.asyncio
async def test_error_response_does_not_cancel() -> None:
    """A failed request has already ended on the engine side."""

    async def fail() -> Any:
        raise ChannelError("Request failed: boom")

    ipc = _IPC(fail)
    result = await _submit(ipc, None)

    await _drain_cancels()
    assert isinstance(result, JSONResponse)
    assert result.status_code == 500
    assert ipc.cancelled == []


class _Clock:
    """Fake clock for the cancel retry window; only the IPC stub advances it."""

    def __init__(self) -> None:
        self.now = 0.0


class _FullIPC:
    """IPC stub at its pending limit for the first `full_for` cancel attempts.

    Each attempt takes `attempt_s` on the fake clock.
    """

    def __init__(self, clock: _Clock, full_for: int, attempt_s: float = 0.0) -> None:
        self._clock = clock
        self._full_for = full_for
        self._attempt_s = attempt_s
        self.attempts = 0
        self.cancelled: list[str] = []

    async def submit(self, request: Any, timeout: float | None = None) -> Any:
        self.attempts += 1
        self._clock.now += self._attempt_s
        if self.attempts <= self._full_for:
            raise RuntimeError("Max pending requests exceeded (100)")
        self.cancelled.append(request.request_id)
        return CancelResponse(id=request.id, found=True)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Retry window on a fake clock, retries without a real pause."""
    fake = _Clock()
    monkeypatch.setattr(cancel_module, "_now", lambda: fake.now)
    monkeypatch.setattr(cancel_module, "CANCEL_RETRY_INTERVAL", 0)
    return fake


@pytest.mark.asyncio
async def test_cancel_retries_while_channel_is_full(clock: _Clock) -> None:
    ipc = _FullIPC(clock, full_for=3)
    lg = MagicMock(spec=Logger)

    cancel_module.send_cancel(lg, ipc, "r1")
    await _drain_cancels()

    assert ipc.cancelled == ["r1"]
    assert ipc.attempts == 4
    lg.warning.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_gives_up_after_retry_window(clock: _Clock) -> None:
    window = cancel_module.CANCEL_RETRY_WINDOW
    ipc = _FullIPC(clock, full_for=10_000, attempt_s=window / 3)
    lg = MagicMock(spec=Logger)

    cancel_module.send_cancel(lg, ipc, "r1")
    await _drain_cancels()

    # Attempts end at window/3, 2/3 and 3/3 of the window; the third reaches it
    assert ipc.attempts == 3
    assert ipc.cancelled == []
    lg.warning.assert_called_once()
    assert lg.warning.call_args[0][0] == "cancel not delivered"


def _generator(ipc: _IPC) -> ChatStreamingGenerator:
    return ChatStreamingGenerator(MagicMock(spec=Logger), "r1", "model", ipc)


@pytest.mark.asyncio
async def test_stream_chunk_timeout_cancels() -> None:
    async def time_out() -> Any:
        raise ChannelTimeoutError("Stream r1 timed out waiting for chunk")

    ipc = _IPC(time_out)
    events = [e async for e in _generator(ipc).stream(_request())]

    await _drain_cancels()
    assert any("timeout" in e for e in events)
    assert ipc.cancelled == ["r1"]


@pytest.mark.asyncio
async def test_stream_client_disconnect_cancels() -> None:
    """Starlette cancels the response task when the client disconnects."""
    ipc = _IPC(_forever)

    async def consume() -> None:
        async for _ in _generator(ipc).stream(_request()):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _drain_cancels()
    assert ipc.cancelled == ["r1"]


@pytest.mark.asyncio
async def test_stream_completed_does_not_cancel() -> None:
    async def final() -> StreamChunk:
        return StreamChunk(id="r1", token="", is_final=True)

    ipc = _IPC(final)
    events = [e async for e in _generator(ipc).stream(_request())]

    await _drain_cancels()
    assert events[-1].strip().endswith("[DONE]")
    assert ipc.cancelled == []
