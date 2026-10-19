# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Abort a single in-flight httpx call from another thread."""

from __future__ import annotations

import socket
import threading
from typing import Any


class RequestAbortedError(RuntimeError):
    """The call was aborted through InFlightConnections.abort()."""


class InFlightConnections:
    """Sockets of in-flight HTTP calls, keyed by request id.

    A worker blocked reading a response can't be woken by closing the httpx
    client from another thread. Shutting its socket down does: the read fails
    at once, and the server sees the client disconnect (vLLM then aborts the
    generation). The socket is captured through httpx's `trace` request
    extension when the TCP connection is made, so the client must not reuse
    pooled connections (no keep-alive); otherwise no connect event fires.

    Usage per call: `extensions=connections.extensions(request_id)` on the
    request, `release(request_id)` once it's finished.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sockets: dict[str, socket.socket] = {}
        # Aborted before (or while) connecting: shut down on connect
        self._aborted: set[str] = set()

    def extensions(self, request_id: str) -> dict[str, Any]:
        """httpx request extensions that register this call's socket."""

        def trace(event: str, info: dict[str, Any]) -> None:
            if event == "connection.connect_tcp.complete":
                self._register(
                    request_id, info["return_value"].get_extra_info("socket")
                )

        return {"trace": trace}

    def _register(self, request_id: str, sock: socket.socket | None) -> None:
        if sock is None:
            return  # Can't abort without socket access
        with self._lock:
            self._sockets[request_id] = sock
            aborted = request_id in self._aborted
        if aborted:
            _shutdown(sock)

    def abort(self, request_id: str) -> None:
        """Abort the call for request_id now, or as soon as it connects."""
        with self._lock:
            self._aborted.add(request_id)
            sock = self._sockets.get(request_id)
        if sock is not None:
            _shutdown(sock)

    def was_aborted(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._aborted

    def release(self, request_id: str) -> None:
        """Forget request_id once its call has finished."""
        with self._lock:
            self._sockets.pop(request_id, None)
            self._aborted.discard(request_id)


def _shutdown(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # Already closed: the call has finished on its own
