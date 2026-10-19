# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for BackendGateMiddleware and SharedBackendState."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_infer.serving.admin.gate import BackendGateMiddleware
from llm_infer.serving.admin.state import BackendState, SharedBackendState

pytestmark = pytest.mark.unit


def _client(state: SharedBackendState) -> TestClient:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat() -> dict[str, str]:
        return {"ok": "chat"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"ok": "health"}

    app.add_middleware(BackendGateMiddleware, state=state, retry_after=42)
    return TestClient(app)


def _state(value: BackendState) -> SharedBackendState:
    shared = SharedBackendState()
    shared.set(value)
    return shared


@pytest.mark.parametrize(
    "value", [BackendState.RELEASING, BackendState.RELEASED, BackendState.LOADING]
)
def test_inference_answers_503_while_released(value: BackendState) -> None:
    r = _client(_state(value)).post("/v1/chat/completions")

    assert r.status_code == 503
    assert r.headers["retry-after"] == "42"
    assert r.json()["error"]["code"] == "unavailable"
    assert value.label in r.json()["error"]["message"]


@pytest.mark.parametrize("value", [BackendState.SERVING, BackendState.INITIALIZING])
def test_inference_passes_otherwise(value: BackendState) -> None:
    r = _client(_state(value)).post("/v1/chat/completions")

    assert r.status_code == 200


def test_health_passes_while_released() -> None:
    r = _client(_state(BackendState.RELEASED)).get("/health")

    assert r.status_code == 200


def test_shared_state_starts_initializing() -> None:
    assert SharedBackendState().get() == BackendState.INITIALIZING
