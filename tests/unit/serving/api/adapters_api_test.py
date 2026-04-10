"""Unit tests for serving/api/adapters.py FastAPI routes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from llm_infer.serving.api.adapters import create_adapter_router
from llm_infer.serving.dispatch.types import (
    AdapterInfo,
    AdapterListResponse,
    AdapterRefreshResponse,
    RequestStatus,
)


def _ok(obj: Any) -> Any:
    """Attach a non-error status to a dataclass response (mirrors real flow)."""
    obj.status = RequestStatus.COMPLETED  # type: ignore[attr-defined]
    return obj


pytestmark = pytest.mark.unit


class _LgInjector(BaseHTTPMiddleware):
    """Inject a logger into request.state, mirroring real middleware."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        request.state.lg = MagicMock(spec=Logger)
        return await call_next(request)


class _StubIPC:
    """Stub IPC channel that records submissions and returns canned responses."""

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
    app.include_router(create_adapter_router(), prefix="/v1")
    return app


# ---------------------------------------------------------------------------
# /v1/adapters (GET)
# ---------------------------------------------------------------------------


def test_list_adapters_empty() -> None:
    ipc = _StubIPC(_ok(AdapterListResponse(id="x", adapters=[])))
    client = TestClient(_make_app(ipc))

    resp = client.get("/v1/adapters")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"adapters": [], "count": 0}
    assert len(ipc.submitted) == 1
    assert ipc.submitted[0].id.startswith("adapter-list-")


def test_list_adapters_with_entries() -> None:
    ipc = _StubIPC(
        _ok(
            AdapterListResponse(
                id="x",
                adapters=[
                    AdapterInfo(
                        key="my-adapter-abc123def456",
                        name="my-adapter",
                        loaded_at="2026-01-01T00:00:00Z",
                        description="test",
                        md5="abc123def456",
                        mtime="2026-01-01T00:00:00Z",
                    ),
                    AdapterInfo(
                        key="other",
                        name="other",
                        loaded_at="2026-01-02T00:00:00Z",
                    ),
                ],
            )
        )
    )
    client = TestClient(_make_app(ipc))

    resp = client.get("/v1/adapters")

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["adapters"][0]["key"] == "my-adapter-abc123def456"
    assert body["adapters"][0]["name"] == "my-adapter"
    assert body["adapters"][0]["description"] == "test"
    assert body["adapters"][1]["description"] is None
    assert body["adapters"][1]["md5"] is None


def test_list_adapters_timeout_returns_504() -> None:
    ipc = _StubIPC(TimeoutError("ipc timeout"))
    client = TestClient(_make_app(ipc))

    resp = client.get("/v1/adapters")

    assert resp.status_code == 504
    body = resp.json()
    assert body["error"]["code"] == "timeout"
    assert body["error"]["type"] == "server_error"


def test_list_adapters_failed_status_returns_500() -> None:
    failed = MagicMock()
    failed.status = RequestStatus.FAILED
    failed.error = "boom"
    ipc = _StubIPC(failed)
    client = TestClient(_make_app(ipc), raise_server_exceptions=False)

    resp = client.get("/v1/adapters")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "boom"


def test_list_adapters_rejected_status_returns_503() -> None:
    rejected = MagicMock()
    rejected.status = RequestStatus.REJECTED
    rejected.error = None
    ipc = _StubIPC(rejected)
    client = TestClient(_make_app(ipc), raise_server_exceptions=False)

    resp = client.get("/v1/adapters")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "Server at capacity"


# ---------------------------------------------------------------------------
# /v1/adapters/refresh (POST)
# ---------------------------------------------------------------------------


def test_refresh_adapters_no_key() -> None:
    ipc = _StubIPC(
        AdapterRefreshResponse(id="x", key=None, adapters_loaded=3, status="scanned")
    )
    client = TestClient(_make_app(ipc))

    resp = client.post("/v1/adapters/refresh")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"key": None, "adapters_loaded": 3, "status": "scanned"}
    assert len(ipc.submitted) == 1
    assert ipc.submitted[0].key is None
    assert ipc.submitted[0].id.startswith("adapter-refresh-")


def test_refresh_adapters_with_key() -> None:
    ipc = _StubIPC(
        AdapterRefreshResponse(
            id="x", key="my-adapter", adapters_loaded=5, status="loaded"
        )
    )
    client = TestClient(_make_app(ipc))

    resp = client.post("/v1/adapters/refresh", params={"key": "my-adapter"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["key"] == "my-adapter"
    assert body["adapters_loaded"] == 5
    assert body["status"] == "loaded"
    assert ipc.submitted[0].key == "my-adapter"


def test_refresh_adapters_timeout_returns_504() -> None:
    ipc = _StubIPC(TimeoutError("ipc timeout"))
    client = TestClient(_make_app(ipc))

    resp = client.post("/v1/adapters/refresh")

    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "timeout"


def test_refresh_adapters_failed_status_returns_500() -> None:
    failed = MagicMock()
    failed.status = RequestStatus.FAILED
    failed.error = None
    ipc = _StubIPC(failed)
    client = TestClient(_make_app(ipc), raise_server_exceptions=False)

    resp = client.post("/v1/adapters/refresh")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal error"
