"""Unit tests for serving/api/openai/router.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from llm_infer.serving.api.openai.router import (
    _build_adapter_fallback_headers,
    _build_adapter_info,
    _build_chat_response,
    _build_completion_response_obj,
    _build_completion_usage,
    _build_embedding_response,
    _convert_tool_calls,
    _create_model_info,
    _create_normalizer,
    _determine_chat_finish_reason,
    _extract_and_separate_thinking,
    _get_think_tags,
    _normalize_response,
    _with_fallback_headers,
    create_openai_router,
)
from llm_infer.serving.dispatch.types import (
    EmbeddingResponse as InternalEmbeddingResponse,
)
from llm_infer.serving.dispatch.types import (
    RequestStatus,
    ResponseAdapterInfo,
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


def _make_app(ipc: _StubIPC, model_name: str = "test-model") -> FastAPI:
    app = FastAPI()
    app.state.ipc_channel = ipc
    app.add_middleware(_LgInjector)
    app.include_router(create_openai_router(model_name), prefix="/v1")
    return app


def _completed_response(
    *,
    result: str = "hello",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    tool_calls: Any = None,
    adapter: Any = None,
) -> InternalResponse:
    return InternalResponse(
        id="x",
        status=RequestStatus.COMPLETED,
        result=result,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_calls=tool_calls,
        adapter=adapter,
    )


# ---------------------------------------------------------------------------
# _build_completion_usage
# ---------------------------------------------------------------------------


def test_build_completion_usage() -> None:
    response = _completed_response(prompt_tokens=10, completion_tokens=5)
    usage = _build_completion_usage(response)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_build_completion_usage_none_tokens() -> None:
    response = InternalResponse(id="x", status=RequestStatus.COMPLETED)
    usage = _build_completion_usage(response)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


# ---------------------------------------------------------------------------
# _create_normalizer / _normalize_response / _get_think_tags
# ---------------------------------------------------------------------------


class TestNormalizer:
    def test_no_model_config(self) -> None:
        assert _create_normalizer(True, None) is None

    def test_think_disabled(self) -> None:
        cfg = MagicMock()
        cfg.think.default = False
        cfg.think.tags_open = ["<think>"]
        cfg.think.tags_close = ["</think>"]
        assert _create_normalizer(False, cfg) is None

    def test_think_enabled(self) -> None:
        cfg = MagicMock()
        cfg.think.default = True
        cfg.think.tags_open = ["<think>", "<thinking>"]
        cfg.think.tags_close = ["</think>", "</thinking>"]
        n = _create_normalizer(True, cfg)
        assert n is not None

    def test_normalize_response_empty(self) -> None:
        assert _normalize_response("", True, None) == ""

    def test_normalize_response_no_normalizer(self) -> None:
        assert _normalize_response("text", True, None) == "text"

    def test_get_think_tags_default(self) -> None:
        open_tags, close_tags = _get_think_tags(None)
        assert "<think>" in open_tags
        assert "</think>" in close_tags

    def test_get_think_tags_from_config(self) -> None:
        cfg = MagicMock()
        cfg.think.tags_open = ["<reason>"]
        cfg.think.tags_close = ["</reason>"]
        open_tags, close_tags = _get_think_tags(cfg)
        assert open_tags == ["<reason>"]
        assert close_tags == ["</reason>"]


# ---------------------------------------------------------------------------
# _extract_and_separate_thinking
# ---------------------------------------------------------------------------


class TestExtractAndSeparateThinking:
    def test_empty_text(self) -> None:
        thinking, content = _extract_and_separate_thinking("", True, None)
        assert thinking is None
        assert content == ""

    def test_no_model_config(self) -> None:
        thinking, content = _extract_and_separate_thinking("text", True, None)
        assert thinking is None
        assert content == "text"

    def test_think_disabled(self) -> None:
        cfg = MagicMock()
        cfg.think.default = False
        thinking, content = _extract_and_separate_thinking(
            "<think>r</think>answer", False, cfg
        )
        assert thinking is None

    def test_think_enabled_extracts(self) -> None:
        cfg = MagicMock()
        cfg.think.default = True
        cfg.think.tags_open = ["<think>"]
        cfg.think.tags_close = ["</think>"]
        thinking, content = _extract_and_separate_thinking(
            "<think>reasoning</think>answer", True, cfg
        )
        assert thinking == "reasoning"
        assert content == "answer"


# ---------------------------------------------------------------------------
# _convert_tool_calls
# ---------------------------------------------------------------------------


class TestConvertToolCalls:
    def test_none(self) -> None:
        assert _convert_tool_calls(None) is None

    def test_empty(self) -> None:
        assert _convert_tool_calls([]) is None

    def test_valid(self) -> None:
        result = _convert_tool_calls(
            [{"function": {"name": "f", "arguments": "{}"}, "id": "tc1"}]
        )
        assert result is not None
        assert len(result) == 1
        assert result[0].id == "tc1"

    def test_skip_malformed(self) -> None:
        result = _convert_tool_calls([{"function": {}}])
        assert result is None


# ---------------------------------------------------------------------------
# _build_chat_response
# ---------------------------------------------------------------------------


def test_build_chat_response() -> None:
    response = _completed_response(prompt_tokens=10, completion_tokens=5)
    chat = _build_chat_response("r1", "model", "hello", None, "stop", response)
    assert chat.id == "r1"
    assert chat.choices[0].message.content == "hello"


# ---------------------------------------------------------------------------
# _determine_chat_finish_reason
# ---------------------------------------------------------------------------


class TestDetermineChatFinishReason:
    def _body(self, max_tokens: int | None = 100) -> Any:
        b = MagicMock()
        b.max_tokens = max_tokens
        b.max_completion_tokens = None
        return b

    def test_max_tokens_reached(self) -> None:
        response = MagicMock(completion_tokens=100)
        result = _determine_chat_finish_reason(response, self._body(100), False)
        assert result == "length"

    def test_eos(self) -> None:
        response = MagicMock(completion_tokens=50)
        result = _determine_chat_finish_reason(response, self._body(100), False)
        assert result == "stop"

    def test_tool_calls(self) -> None:
        response = MagicMock(completion_tokens=50)
        result = _determine_chat_finish_reason(response, self._body(100), True)
        assert result == "tool_calls"


# ---------------------------------------------------------------------------
# _build_adapter_info / _build_adapter_fallback_headers / _with_fallback_headers
# ---------------------------------------------------------------------------


class TestAdapterInfoHelpers:
    def test_build_adapter_info_none(self) -> None:
        response = _completed_response()
        assert _build_adapter_info(response) is None

    def test_build_adapter_info_present(self) -> None:
        adapter = ResponseAdapterInfo(
            requested="x", actual="x", fallback=False, mtime="t", md5="m"
        )
        response = _completed_response(adapter=adapter)
        info = _build_adapter_info(response)
        assert info is not None
        assert info.requested == "x"

    def test_fallback_headers_none(self) -> None:
        response = _completed_response()
        assert _build_adapter_fallback_headers(response) == {}

    def test_fallback_headers_present(self) -> None:
        adapter = ResponseAdapterInfo(requested="x", fallback=True)
        response = _completed_response(adapter=adapter)
        headers = _build_adapter_fallback_headers(response)
        assert headers["X-Adapter-Fallback"] == "true"
        assert headers["X-Adapter-Requested"] == "x"

    def test_with_fallback_headers_no_fallback(self) -> None:
        from llm_infer.schemas.openai import ModelInfo

        m = ModelInfo(id="test", created=0, owned_by="local")
        response = _completed_response()
        result = _with_fallback_headers(m, response)
        assert result is m  # Same object

    def test_with_fallback_headers_with_fallback(self) -> None:
        from fastapi.responses import JSONResponse

        from llm_infer.schemas.openai import ModelInfo

        m = ModelInfo(id="test", created=0, owned_by="local")
        adapter = ResponseAdapterInfo(requested="x", fallback=True)
        response = _completed_response(adapter=adapter)
        result = _with_fallback_headers(m, response)
        assert isinstance(result, JSONResponse)


# ---------------------------------------------------------------------------
# _build_completion_response_obj
# ---------------------------------------------------------------------------


class TestBuildCompletionResponseObj:
    def _body(
        self, *, prompt: str = "hi", echo: bool = False, max_tokens: int = 100
    ) -> Any:
        b = MagicMock()
        b.prompt = prompt
        b.echo = echo
        b.max_tokens = max_tokens
        return b

    def test_basic(self) -> None:
        response = _completed_response(result="result")
        obj = _build_completion_response_obj("r1", self._body(), "model", response)
        assert obj.choices[0].text == "result"

    def test_with_echo(self) -> None:
        response = _completed_response(result="generated")
        obj = _build_completion_response_obj(
            "r1", self._body(prompt="prefix:", echo=True), "model", response
        )
        assert obj.choices[0].text == "prefix:generated"

    def test_with_echo_list_prompt(self) -> None:
        response = _completed_response(result="gen")
        obj = _build_completion_response_obj(
            "r1", self._body(prompt=["a", "b"], echo=True), "model", response
        )
        assert obj.choices[0].text == "agen"

    def test_max_tokens_reached(self) -> None:
        response = _completed_response(completion_tokens=100)
        obj = _build_completion_response_obj(
            "r1", self._body(max_tokens=100), "model", response
        )
        assert obj.choices[0].finish_reason == "length"


# ---------------------------------------------------------------------------
# _create_model_info
# ---------------------------------------------------------------------------


def test_create_model_info() -> None:
    m = _create_model_info("test-model")
    assert m.id == "test-model"
    assert m.owned_by == "local"


# ---------------------------------------------------------------------------
# _build_embedding_response
# ---------------------------------------------------------------------------


def test_build_embedding_response() -> None:
    response = MagicMock(embeddings=[[0.1, 0.2], [0.3, 0.4]], total_tokens=10)
    result = _build_embedding_response(response, "test-model")
    assert len(result.data) == 2
    assert result.data[0].embedding == [0.1, 0.2]
    assert result.usage.prompt_tokens == 10
    assert result.model == "test-model"


def test_build_embedding_response_empty() -> None:
    response = MagicMock(embeddings=None, total_tokens=0)
    result = _build_embedding_response(response, "test-model")
    assert result.data == []


# ---------------------------------------------------------------------------
# /v1/models endpoints
# ---------------------------------------------------------------------------


class TestModelsEndpoints:
    def test_list_models(self) -> None:
        ipc = _StubIPC(MagicMock())
        client = TestClient(_make_app(ipc, model_name="qwen-7b"))
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "qwen-7b"

    def test_get_model_found(self) -> None:
        ipc = _StubIPC(MagicMock())
        client = TestClient(_make_app(ipc, model_name="qwen-7b"))
        resp = client.get("/v1/models/qwen-7b")
        assert resp.status_code == 200

    def test_get_model_not_found(self) -> None:
        ipc = _StubIPC(MagicMock())
        client = TestClient(_make_app(ipc, model_name="qwen-7b"))
        resp = client.get("/v1/models/other-model")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /v1/chat/completions endpoint
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_basic_non_streaming(self) -> None:
        response = _completed_response(result="hello")
        ipc = _StubIPC(response)
        client = TestClient(_make_app(ipc))
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "hello"

    def test_with_max_tokens(self) -> None:
        response = _completed_response()
        ipc = _StubIPC(response)
        client = TestClient(_make_app(ipc))
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200

    def test_with_tool_calls(self) -> None:
        response = _completed_response(
            tool_calls=[{"function": {"name": "f", "arguments": "{}"}, "id": "tc1"}]
        )
        ipc = _StubIPC(response)
        client = TestClient(_make_app(ipc))
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    def test_timeout(self) -> None:
        ipc = _StubIPC(TimeoutError("timeout"))
        client = TestClient(_make_app(ipc))
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 504


# ---------------------------------------------------------------------------
# /v1/completions endpoint
# ---------------------------------------------------------------------------


class TestCompletions:
    def test_basic(self) -> None:
        response = _completed_response(result="generated")
        ipc = _StubIPC(response)
        client = TestClient(_make_app(ipc))
        resp = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hello", "max_tokens": 50},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["text"] == "generated"

    def test_with_echo(self) -> None:
        response = _completed_response(result=" generated")
        ipc = _StubIPC(response)
        client = TestClient(_make_app(ipc))
        resp = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "prompt:", "echo": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["text"] == "prompt: generated"


# ---------------------------------------------------------------------------
# /v1/embeddings endpoint
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_single_input(self) -> None:
        response = InternalEmbeddingResponse(
            id="x",
            status=RequestStatus.COMPLETED,
            embeddings=[[0.1, 0.2, 0.3]],
            total_tokens=5,
        )
        ipc = _StubIPC(response)
        client = TestClient(_make_app(ipc))
        resp = client.post("/v1/embeddings", json={"model": "test", "input": "hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 1

    def test_list_input(self) -> None:
        response = InternalEmbeddingResponse(
            id="x",
            status=RequestStatus.COMPLETED,
            embeddings=[[0.1], [0.2]],
            total_tokens=10,
        )
        ipc = _StubIPC(response)
        client = TestClient(_make_app(ipc))
        resp = client.post(
            "/v1/embeddings", json={"model": "test", "input": ["a", "b"]}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 2

    def test_timeout(self) -> None:
        ipc = _StubIPC(TimeoutError("timeout"))
        client = TestClient(_make_app(ipc))
        resp = client.post("/v1/embeddings", json={"model": "test", "input": "hi"})
        assert resp.status_code == 504
