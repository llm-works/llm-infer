# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""serving/api/trace.py middleware through TestClient (requests run on its portal
thread)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from appinfra.log import Logger
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from llm_infer.serving.api.trace import TraceMiddleware


def _make_app(lg: Logger | None = None) -> FastAPI:
    app = FastAPI()
    if lg is not None:
        app.state.lg = lg
    app.add_middleware(TraceMiddleware)

    @app.get("/v1/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "yes"}

    @app.post("/v1/echo")
    async def echo(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    @app.get("/v1/stream")
    async def stream() -> StreamingResponse:
        async def gen():
            yield b"hello "
            yield "world"

        return StreamingResponse(gen(), media_type="text/plain")

    @app.get("/v1/sse")
    async def sse() -> StreamingResponse:
        async def gen():
            yield "data: 1\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_trace_skipped_when_no_logger() -> None:
    # No logger on app.state -> middleware passes through unchanged
    client = TestClient(_make_app(lg=None))
    resp = client.get("/v1/ping")
    assert resp.status_code == 200
    assert resp.json() == {"ok": "yes"}


def test_trace_skipped_for_non_v1_paths() -> None:
    lg = MagicMock(spec=Logger)
    client = TestClient(_make_app(lg=lg))
    resp = client.get("/health")
    assert resp.status_code == 200
    # No trace calls for non-/v1 path
    assert not lg.trace.called


def test_trace_logs_request_body() -> None:
    lg = MagicMock(spec=Logger)
    client = TestClient(_make_app(lg=lg))
    resp = client.post("/v1/echo", json={"hello": "world"})
    assert resp.status_code == 200
    # Request body should be logged at trace level
    request_calls = [
        c for c in lg.trace.call_args_list if c.args and c.args[0] == "request"
    ]
    assert len(request_calls) == 1
    assert "hello" in request_calls[0].kwargs["extra"]["body"]


def test_trace_skips_empty_request_body() -> None:
    lg = MagicMock(spec=Logger)
    client = TestClient(_make_app(lg=lg))
    resp = client.get("/v1/ping")
    assert resp.status_code == 200
    # GET with no body -> no "request" trace
    request_calls = [
        c for c in lg.trace.call_args_list if c.args and c.args[0] == "request"
    ]
    assert len(request_calls) == 0


def test_trace_streaming_response_passes_through() -> None:
    """Streaming responses still flow correctly through the middleware."""
    lg = MagicMock(spec=Logger)
    client = TestClient(_make_app(lg=lg))
    resp = client.get("/v1/stream")
    assert resp.status_code == 200
    assert resp.text == "hello world"


def test_trace_sse_passes_through() -> None:
    """SSE responses are not intercepted (preserves streaming)."""
    lg = MagicMock(spec=Logger)
    client = TestClient(_make_app(lg=lg))
    resp = client.get("/v1/sse")
    assert resp.status_code == 200


def test_trace_request_body_decode_error_logged() -> None:
    """If request body cannot be read/decoded, the error is logged as a warning."""
    lg = MagicMock(spec=Logger)
    app = FastAPI()
    app.state.lg = lg
    app.add_middleware(TraceMiddleware)

    @app.post("/v1/binary")
    async def binary() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    # Send invalid utf-8 bytes — body.decode() will raise inside _log_request
    resp = client.post(
        "/v1/binary",
        content=b"\xff\xfe\xfd",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    assert lg.warning.called
