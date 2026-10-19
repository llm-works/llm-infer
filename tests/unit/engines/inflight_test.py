# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for InFlightConnections (abort one httpx call from another thread)."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from llm_infer.engines.inflight import InFlightConnections, RequestAbortedError

pytestmark = pytest.mark.unit


class SilentServer:
    """Accepts connections, reads the request, never answers.

    Records whether each connection saw the client close it (EOF).
    """

    def __init__(self) -> None:
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen()
        self.url = f"http://127.0.0.1:{self._srv.getsockname()[1]}/x"
        self.received = threading.Event()
        self.client_closed = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._hold, args=(conn,), daemon=True).start()

    def _hold(self, conn: socket.socket) -> None:
        conn.recv(65536)
        self.received.set()
        # Drain the rest of the request (headers and body may come in separate
        # segments) until EOF: the client closed its side
        try:
            while conn.recv(65536):
                pass
        except OSError:
            pass  # reset counts as closed too
        self.client_closed.set()

    def close(self) -> None:
        self._srv.close()


@pytest.fixture
def server() -> Iterator[SilentServer]:
    s = SilentServer()
    yield s
    s.close()


def _client() -> httpx.Client:
    return httpx.Client(timeout=30.0, limits=httpx.Limits(max_keepalive_connections=0))


def _call_in_thread(
    client: httpx.Client, url: str, extensions: dict[str, Any]
) -> tuple[threading.Thread, dict[str, Any]]:
    result: dict[str, Any] = {}

    def call() -> None:
        try:
            client.post(url, json={}, extensions=extensions)
            result["outcome"] = "returned"
        except httpx.HTTPError as e:
            result["error"] = e

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    return thread, result


def test_abort_ends_blocked_call_and_server_sees_disconnect(
    server: SilentServer,
) -> None:
    connections = InFlightConnections()
    thread, result = _call_in_thread(
        _client(), server.url, connections.extensions("r1")
    )
    assert server.received.wait(5)

    start = time.monotonic()
    connections.abort("r1")
    thread.join(timeout=5)

    assert time.monotonic() - start < 2.0
    assert isinstance(result.get("error"), httpx.HTTPError)
    assert connections.was_aborted("r1")
    assert server.client_closed.wait(5)


def test_abort_before_connect_aborts_on_connect(server: SilentServer) -> None:
    connections = InFlightConnections()
    connections.abort("r1")

    thread, result = _call_in_thread(
        _client(), server.url, connections.extensions("r1")
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert isinstance(result.get("error"), httpx.HTTPError)


def test_abort_other_request_leaves_call_running(server: SilentServer) -> None:
    connections = InFlightConnections()
    thread, result = _call_in_thread(
        _client(), server.url, connections.extensions("r1")
    )
    assert server.received.wait(5)

    connections.abort("r2")
    thread.join(timeout=0.5)

    assert thread.is_alive()
    assert result == {}
    connections.abort("r1")  # cleanup
    thread.join(timeout=5)


def test_release_forgets_request() -> None:
    connections = InFlightConnections()
    connections.abort("r1")

    connections.release("r1")

    assert not connections.was_aborted("r1")


# ---------------------------------------------------------------------------
# VLLMServerEngine: abort() on blocking and streaming calls
# ---------------------------------------------------------------------------


def _engine(server: SilentServer) -> Any:
    from llm_infer.engines.vllm_server import VLLMServerEngine

    engine = object.__new__(VLLMServerEngine)
    engine._lg = MagicMock()
    engine._connections = InFlightConnections()
    engine._client = httpx.Client(
        base_url=server.url.rsplit("/", 1)[0],
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=0),
    )
    return engine


def _run(fn: Any) -> tuple[threading.Thread, dict[str, Any]]:
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn()
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, result


def test_engine_abort_blocking_call(server: SilentServer) -> None:
    engine = _engine(server)
    thread, result = _run(lambda: engine._post_chat_completions({}, "r1"))
    assert server.received.wait(5)

    engine.abort("r1")
    thread.join(timeout=5)

    assert isinstance(result.get("error"), RequestAbortedError)
    engine._lg.error.assert_not_called()  # an abort is not a failure
    assert not engine._connections.was_aborted("r1")  # released
    assert server.client_closed.wait(5)


def test_engine_abort_streaming_call(server: SilentServer) -> None:
    from llm_infer.engines.vllm_server import VLLMServerStreamingIterator

    engine = _engine(server)
    stream = VLLMServerStreamingIterator(
        engine._lg,
        engine._client,
        "/v1/chat/completions",
        {},
        None,
        abort_scope=(engine._connections, "r1"),
    )
    thread, result = _run(lambda: next(stream))
    assert server.received.wait(5)

    engine.abort("r1")
    thread.join(timeout=5)

    assert isinstance(result.get("error"), RequestAbortedError)
    assert not engine._connections.was_aborted("r1")  # released
    assert server.client_closed.wait(5)
