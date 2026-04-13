"""Unit tests for serving/api/routes.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from llm_infer.serving.api.routes import (
    create_health_handler,
    create_routes,
    health_handler,
)
from llm_infer.serving.dispatch.types import (
    MetricsResponse,
    RequestStatus,
)
from llm_infer.serving.dispatch.types import (
    Response as InternalResponse,
)

pytestmark = pytest.mark.unit


class _LgInjector(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        request.state.lg = MagicMock(spec=Logger)
        return await call_next(request)


class _StubIPC:
    def __init__(self, response: Any | Exception) -> None:
        self.response = response
        self.submitted: list[Any] = []

    async def submit(self, request: Any) -> Any:
        self.submitted.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _make_app(ipc: _StubIPC) -> FastAPI:
    app = FastAPI()
    app.state.ipc_channel = ipc
    app.add_middleware(_LgInjector)
    app.include_router(create_routes("test-model"))
    return app


# ---------------------------------------------------------------------------
# health_handler / create_health_handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_health_handler() -> None:
    resp = await health_handler()
    assert resp.status == "ok"


@pytest.mark.asyncio
async def test_create_health_handler_no_ready_flag() -> None:
    h = create_health_handler(None)
    resp = await h()
    assert resp.status == "ok"


@pytest.mark.asyncio
async def test_create_health_handler_initializing() -> None:
    flag = MagicMock()
    flag.value = False
    h = create_health_handler(flag)
    resp = await h()
    assert resp.status == "initializing"


@pytest.mark.asyncio
async def test_create_health_handler_ready() -> None:
    flag = MagicMock()
    flag.value = True
    h = create_health_handler(flag)
    resp = await h()
    assert resp.status == "ok"


# ---------------------------------------------------------------------------
# /generate endpoint
# ---------------------------------------------------------------------------


def test_generate_success() -> None:
    response = InternalResponse(
        id="x",
        status=RequestStatus.COMPLETED,
        result="generated text",
        prompt_tokens=5,
        completion_tokens=3,
    )
    ipc = _StubIPC(response)
    client = TestClient(_make_app(ipc))
    resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "generated text"
    assert body["prompt_tokens"] == 5
    assert body["completion_tokens"] == 3


def test_generate_with_full_params() -> None:
    response = InternalResponse(
        id="x",
        status=RequestStatus.COMPLETED,
        result="x",
        prompt_tokens=1,
        completion_tokens=1,
    )
    ipc = _StubIPC(response)
    client = TestClient(_make_app(ipc))
    resp = client.post(
        "/generate",
        json={
            "prompt": "hi",
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.2,
            "use_chat_template": True,
        },
    )
    assert resp.status_code == 200
    submitted = ipc.submitted[0]
    assert submitted.max_tokens == 50
    assert submitted.temperature == 0.7
    assert submitted.use_chat_template is True


def test_generate_timeout_returns_504() -> None:
    ipc = _StubIPC(TimeoutError("ipc timeout"))
    client = TestClient(_make_app(ipc))
    resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 504


def test_generate_failed_status_returns_500() -> None:
    failed = InternalResponse(id="x", status=RequestStatus.FAILED, error="boom")
    ipc = _StubIPC(failed)
    client = TestClient(_make_app(ipc), raise_server_exceptions=False)
    resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 500


def test_generate_handles_none_result() -> None:
    """Result/tokens default to 0/empty when None."""
    response = InternalResponse(id="x", status=RequestStatus.COMPLETED)
    ipc = _StubIPC(response)
    client = TestClient(_make_app(ipc))
    resp = client.post("/generate", json={"prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json() == {"text": "", "prompt_tokens": 0, "completion_tokens": 0}


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------


def _metrics_response() -> MetricsResponse:
    return MetricsResponse(
        id="x",
        gpu_allocated_bytes=1024,
        gpu_reserved_bytes=2048,
        gpu_peak_bytes=4096,
        kv_cache_bytes=512,
        active_sequences=2,
        total_sequence_tokens=100,
        pending_requests=3,
    )


def test_metrics_success() -> None:
    response = _metrics_response()
    response.status = RequestStatus.COMPLETED  # type: ignore[attr-defined]
    ipc = _StubIPC(response)
    client = TestClient(_make_app(ipc))
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "gpu" in body
    assert body["pending_requests"] == 3


def test_metrics_with_reset_peak() -> None:
    response = _metrics_response()
    response.status = RequestStatus.COMPLETED  # type: ignore[attr-defined]
    ipc = _StubIPC(response)
    client = TestClient(_make_app(ipc))
    resp = client.get("/metrics", params={"reset_peak": "true"})
    assert resp.status_code == 200
    assert ipc.submitted[0].reset_peak is True


def test_metrics_timeout_returns_504() -> None:
    ipc = _StubIPC(TimeoutError("ipc timeout"))
    client = TestClient(_make_app(ipc))
    resp = client.get("/metrics")
    assert resp.status_code == 504


def test_metrics_failed_status_returns_500() -> None:
    failed = MagicMock()
    failed.status = RequestStatus.FAILED
    failed.error = "boom"
    ipc = _StubIPC(failed)
    client = TestClient(_make_app(ipc), raise_server_exceptions=False)
    resp = client.get("/metrics")
    assert resp.status_code == 500
