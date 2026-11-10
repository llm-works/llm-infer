# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""OpenAI router endpoints through TestClient (requests run on its portal thread)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from appinfra.log import Logger
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from llm_infer.serving.api.openai.router import (
    OpenAIRouterConfig,
    create_openai_router,
)
from llm_infer.serving.dispatch.types import (
    EmbeddingResponse as InternalEmbeddingResponse,
)
from llm_infer.serving.dispatch.types import RequestStatus
from llm_infer.serving.dispatch.types import Response as InternalResponse


class _LgInjector(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        request.state.lg = MagicMock(spec=Logger)
        return await call_next(request)


class _StubIPC:
    def __init__(self, response: Any | Exception) -> None:
        self.response = response
        self.submitted: list[Any] = []

    async def submit(self, request: Any, timeout: float | None = None) -> Any:
        self.submitted.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _client(response: Any | Exception, model_name: str = "test-model") -> TestClient:
    app = FastAPI()
    app.state.ipc_channel = _StubIPC(response)
    app.add_middleware(_LgInjector)
    app.include_router(
        create_openai_router(OpenAIRouterConfig(model_name=model_name)), prefix="/v1"
    )
    return TestClient(app)


def _completed(**fields: Any) -> InternalResponse:
    values: dict[str, Any] = {"result": "hello", "prompt_tokens": 5}
    values.update(fields)
    return InternalResponse(
        id="x", status=RequestStatus.COMPLETED, completion_tokens=3, **values
    )


_CHAT = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}


class TestModelsEndpoints:
    def test_list_models(self) -> None:
        resp = _client(MagicMock(), model_name="qwen-7b").get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "qwen-7b"

    def test_get_model_found(self) -> None:
        client = _client(MagicMock(), model_name="qwen-7b")
        assert client.get("/v1/models/qwen-7b").status_code == 200

    def test_get_model_not_found(self) -> None:
        client = _client(MagicMock(), model_name="qwen-7b")
        assert client.get("/v1/models/other-model").status_code == 404


class TestChatCompletions:
    def test_basic_non_streaming(self) -> None:
        resp = _client(_completed()).post("/v1/chat/completions", json=_CHAT)
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hello"

    def test_with_max_tokens(self) -> None:
        body = {**_CHAT, "max_tokens": 50}
        resp = _client(_completed()).post("/v1/chat/completions", json=body)
        assert resp.status_code == 200

    def test_with_tool_calls(self) -> None:
        tool_calls = [{"function": {"name": "f", "arguments": "{}"}, "id": "tc1"}]
        client = _client(_completed(tool_calls=tool_calls))
        resp = client.post("/v1/chat/completions", json=_CHAT)
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"

    def test_timeout(self) -> None:
        client = _client(TimeoutError("timeout"))
        assert client.post("/v1/chat/completions", json=_CHAT).status_code == 504


class TestCompletions:
    def test_basic(self) -> None:
        client = _client(_completed(result="generated"))
        resp = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hello", "max_tokens": 50},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["text"] == "generated"

    def test_with_echo(self) -> None:
        client = _client(_completed(result=" generated"))
        resp = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "prompt:", "echo": True},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["text"] == "prompt: generated"


def _embedding(embeddings: list[list[float]], total_tokens: int) -> Any:
    return InternalEmbeddingResponse(
        id="x",
        status=RequestStatus.COMPLETED,
        embeddings=embeddings,
        total_tokens=total_tokens,
    )


class TestEmbeddings:
    def test_single_input(self) -> None:
        client = _client(_embedding([[0.1, 0.2, 0.3]], 5))
        resp = client.post("/v1/embeddings", json={"model": "test", "input": "hello"})
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 1

    def test_list_input(self) -> None:
        client = _client(_embedding([[0.1], [0.2]], 10))
        resp = client.post(
            "/v1/embeddings", json={"model": "test", "input": ["a", "b"]}
        )
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 2

    def test_timeout(self) -> None:
        client = _client(TimeoutError("timeout"))
        resp = client.post("/v1/embeddings", json={"model": "test", "input": "hi"})
        assert resp.status_code == 504
