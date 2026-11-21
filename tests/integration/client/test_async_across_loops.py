# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""One client used from several threads, each running its own event loop."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import ChatRequest, Factory, LLMClient
from llm_infer.client.backends import OpenAICompatibleBackend

T = TypeVar("T")

USER = [{"role": "user", "content": "Hi"}]
WAIT_S = 10.0
"""Failure deadline for waits on other threads; passing runs never reach it."""

_COMPLETION = json.dumps(
    {
        "id": "c1",
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }
).encode()


@dataclass
class _Server:
    """A local completion server; requests wait for ``gate`` before answering."""

    base_url: str = ""
    gate: threading.Event = field(default_factory=threading.Event)
    arrived: threading.Event = field(default_factory=threading.Event)

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Answer requests on one kept-alive connection until the client leaves."""
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                await reader.readexactly(_content_length(head))
                self.arrived.set()
                if not self.gate.is_set():
                    await asyncio.to_thread(self.gate.wait, WAIT_S)
                writer.write(_RESPONSE)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


_RESPONSE = (
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
    b"Content-Length: %d\r\n\r\n" % len(_COMPLETION) + _COMPLETION
)


def _content_length(head: bytes) -> int:
    for line in head.split(b"\r\n"):
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            return int(value)
    return 0


@contextmanager
def _loop_thread() -> Iterator[asyncio.AbstractEventLoop]:
    """An event loop running in its own thread until the block ends."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        _on(loop, _cancel_tasks())
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


async def _cancel_tasks() -> None:
    """Cancel every other task of the running loop and wait for them to end."""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
def server() -> Iterator[_Server]:
    state = _Server()
    state.gate.set()
    with _loop_thread() as loop:
        listener = _on(loop, asyncio.start_server(state.handle, "127.0.0.1", 0))
        port = listener.sockets[0].getsockname()[1]
        state.base_url = f"http://127.0.0.1:{port}/v1"
        try:
            yield state
        finally:
            state.gate.set()
            loop.call_soon_threadsafe(listener.close)


@pytest.fixture
def client(server: _Server) -> LLMClient:
    return Factory(MagicMock(spec=Logger)).openai(
        base_url=server.base_url, default_model="m", timeout=WAIT_S
    )


@pytest.fixture
def backend(server: _Server) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        MagicMock(spec=Logger), "test", default_model="m", base_url=server.base_url
    )


@pytest.fixture
def worker_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """An event loop running in its own thread until the test ends."""
    with _loop_thread() as loop:
        yield loop


def _in_thread(fn: Callable[[], Any]) -> Any:
    """Run fn in a new thread and return its result, or raise its exception."""
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result["value"] = fn()
        except BaseException as e:  # re-raised in the calling thread
            result["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _on(loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, T]) -> T:
    """Run coro on a loop of another thread and return its result."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(WAIT_S)


def test_second_thread_loop_after_first_loop_closed(client: LLMClient) -> None:
    # asyncio.run closes the first thread's loop; its pooled connection must not
    # be handed to the second thread's loop.
    first = _in_thread(lambda: asyncio.run(client.chat_async(USER)))
    second = _in_thread(lambda: asyncio.run(client.chat_async(USER)))
    assert (first.content, second.content) == ("ok", "ok")


def test_loops_in_threads_call_at_once(client: LLMClient) -> None:
    # Rounds leave idle pooled connections that a later round could take from
    # another loop's pool.
    threads, rounds = 4, 3
    barrier = threading.Barrier(threads)
    answers: list[str] = []
    lock = threading.Lock()

    async def calls() -> list[str]:
        contents: list[str] = []
        for _ in range(rounds):
            barrier.wait(WAIT_S)  # every thread's loop starts the round together
            responses = await asyncio.gather(
                *(client.chat_async(USER) for _ in range(3))
            )
            contents.extend(r.content for r in responses)
        return contents

    errors: list[BaseException] = []

    def run() -> None:
        try:
            contents = asyncio.run(calls())
        except BaseException as e:  # reported below, from the test's thread
            barrier.abort()  # release the other threads
            with lock:
                errors.append(e)
            return
        with lock:
            answers.extend(contents)

    # Daemon threads: a worker stuck on another loop's connection fails below.
    workers = [threading.Thread(target=run, daemon=True) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(WAIT_S)
    assert not any(t.is_alive() for t in workers), "a worker thread is stuck"
    first_error = next(
        (e for e in errors if not isinstance(e, threading.BrokenBarrierError)), None
    )
    assert first_error is None, repr(first_error)
    assert answers == ["ok"] * threads * rounds * 3


def test_aclose_closes_client_of_running_loop_in_other_thread(
    backend: OpenAICompatibleBackend, worker_loop: asyncio.AbstractEventLoop
) -> None:
    request = ChatRequest(messages=USER)
    assert _on(worker_loop, backend.chat_async(request)).content == "ok"

    async def worker_client() -> Any:
        return backend._get_async_client()

    async def closed_on_worker() -> bool:
        # Queued after aclose's close task is started on this loop.
        await asyncio.gather(*backend._async_clients._closing)
        return bool(other.is_closed)

    other = _on(worker_loop, worker_client())
    asyncio.run(backend.aclose())
    assert _on(worker_loop, closed_on_worker())
    assert len(backend._async_clients) == 0


def test_aclose_waits_for_request_on_other_thread(
    server: _Server,
    backend: OpenAICompatibleBackend,
    worker_loop: asyncio.AbstractEventLoop,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    draining = threading.Event()
    drain = backend._drain_async_requests

    async def signalled_drain() -> None:
        task = asyncio.ensure_future(drain())
        await asyncio.sleep(0)  # drain runs up to its wait
        draining.set()
        await task

    monkeypatch.setattr(backend, "_drain_async_requests", signalled_drain)
    server.gate.clear()
    future = asyncio.run_coroutine_threadsafe(
        backend.chat_async(ChatRequest(messages=USER)), worker_loop
    )
    future.add_done_callback(lambda _: order.append("answered"))
    assert server.arrived.wait(WAIT_S)

    def close() -> None:
        asyncio.run(backend.aclose())
        order.append("closed")

    closer = threading.Thread(target=close, daemon=True)  # a hung close fails below
    closer.start()
    assert draining.wait(WAIT_S)
    server.gate.set()
    assert future.result(WAIT_S).content == "ok"
    closer.join(WAIT_S)
    assert not closer.is_alive()
    assert order == ["answered", "closed"]
