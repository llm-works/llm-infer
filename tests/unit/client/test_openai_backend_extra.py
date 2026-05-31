"""Additional OpenAI backend tests covering error paths and missing methods.

Existing test_openai_backend.py covers init, chat success/error, async, basic
streaming, helpers, and resource management. This file fills:
- list_models() and its error paths
- _execute_sync/async error paths (5 exception types each)
- _execute_stream_sync/async error paths
- chat_stream_async()
- _parse_sse_line edge cases
- _parse_adapter_info / _parse_finish_reason / _parse_usage
- _StreamState tool_call buffering
- Streaming with adapter/usage chunks
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from appinfra.log import Logger

from llm_infer.client.backends.providers.openai import (
    OpenAICompatibleBackend,
    _parse_adapter_info,
    _parse_finish_reason,
    _parse_usage,
    _StreamState,
)
from llm_infer.client.errors import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_infer.client.types import ChatRequest
from llm_infer.schemas.openai import FinishReason

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    return MagicMock(spec=Logger)


def _backend(mock_lg: Logger, **kwargs: Any) -> OpenAICompatibleBackend:
    backend = OpenAICompatibleBackend(mock_lg, "test", **kwargs)
    return backend


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestParseFinishReason:
    def test_none(self) -> None:
        assert _parse_finish_reason(None) is None

    def test_known_value(self) -> None:
        assert _parse_finish_reason("stop") == FinishReason.STOP

    def test_unknown_value(self) -> None:
        assert _parse_finish_reason("invalid") == "invalid"

    def test_gemini_function_call_filter(self) -> None:
        value = "function_call_filter: MALFORMED_FUNCTION_CALL"
        assert _parse_finish_reason(value) == value


class TestParseUsage:
    def test_none(self) -> None:
        assert _parse_usage(None) is None

    def test_full(self) -> None:
        result = _parse_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert result is not None
        assert result.prompt_tokens == 10
        assert result.total_tokens == 15

    def test_missing_fields(self) -> None:
        result = _parse_usage({})
        assert result is not None
        assert result.prompt_tokens == 0


class TestParseAdapterInfo:
    def test_none(self) -> None:
        assert _parse_adapter_info(None) is None

    def test_full(self) -> None:
        result = _parse_adapter_info(
            {
                "requested": "x",
                "actual": "x",
                "fallback": False,
                "mtime": "t",
                "md5": "m",
            }
        )
        assert result is not None
        assert result.requested == "x"

    def test_default_fallback_false(self) -> None:
        result = _parse_adapter_info({"requested": "x"})
        assert result is not None
        assert result.fallback is False


# ---------------------------------------------------------------------------
# list_models() and error paths
# ---------------------------------------------------------------------------


class TestListModels:
    def test_success(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
        mock_resp.raise_for_status = MagicMock()
        backend._client.get = MagicMock(return_value=mock_resp)

        result = backend.list_models()
        assert result == ["model1", "model2"]
        backend.close()

    def test_connection_error(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        backend._client.get = MagicMock(
            side_effect=httpx.ConnectError("connection failed")
        )
        with pytest.raises(BackendUnavailableError):
            backend.list_models()
        backend.close()

    def test_timeout(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        backend._client.get = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(BackendTimeoutError):
            backend.list_models()
        backend.close()

    def test_status_error(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal error"
        backend._client.get = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "boom", request=MagicMock(), response=mock_resp
            )
        )
        with pytest.raises(BackendRequestError) as exc_info:
            backend.list_models()
        assert exc_info.value.status_code == 500
        backend.close()

    def test_request_error(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        backend._client.get = MagicMock(
            side_effect=httpx.RequestError("transport failure")
        )
        with pytest.raises(BackendRequestError):
            backend.list_models()
        backend.close()

    def test_invalid_json(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        mock_resp = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "{", 0)
        mock_resp.raise_for_status = MagicMock()
        backend._client.get = MagicMock(return_value=mock_resp)
        with pytest.raises(BackendRequestError):
            backend.list_models()
        backend.close()


# ---------------------------------------------------------------------------
# _execute_sync error paths
# ---------------------------------------------------------------------------


class TestExecuteSyncErrors:
    def test_connection_error(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        backend._client.post = MagicMock(side_effect=httpx.ConnectError("conn failed"))
        with pytest.raises(BackendUnavailableError):
            backend._execute_sync("http://x", {})
        backend.close()

    def test_timeout(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        backend._client.post = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(BackendTimeoutError):
            backend._execute_sync("http://x", {})
        backend.close()

    def test_status_error(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "boom"
        backend._client.post = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "boom", request=MagicMock(), response=mock_resp
            )
        )
        with pytest.raises(BackendRequestError):
            backend._execute_sync("http://x", {})
        backend.close()

    def test_request_error(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        backend._client.post = MagicMock(side_effect=httpx.RequestError("transport"))
        with pytest.raises(BackendRequestError):
            backend._execute_sync("http://x", {})
        backend.close()

    def test_invalid_json(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        mock_resp = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "{", 0)
        mock_resp.raise_for_status = MagicMock()
        backend._client.post = MagicMock(return_value=mock_resp)
        with pytest.raises(BackendRequestError):
            backend._execute_sync("http://x", {})
        backend.close()


# ---------------------------------------------------------------------------
# _execute_async error paths
# ---------------------------------------------------------------------------


class TestExecuteAsyncErrors:
    def _setup_async(
        self, mock_lg: Logger, side_effect: Any
    ) -> OpenAICompatibleBackend:
        backend = _backend(mock_lg)
        async_client = backend._get_async_client()

        async def fake_post(*args: Any, **kwargs: Any) -> Any:
            raise side_effect

        async_client.post = fake_post
        return backend

    def test_connection_error(self, mock_lg: Logger) -> None:
        backend = self._setup_async(mock_lg, httpx.ConnectError("boom"))
        with pytest.raises(BackendUnavailableError):
            asyncio.run(backend._execute_async("http://x", {}))

    def test_timeout(self, mock_lg: Logger) -> None:
        backend = self._setup_async(mock_lg, httpx.TimeoutException("boom"))
        with pytest.raises(BackendTimeoutError):
            asyncio.run(backend._execute_async("http://x", {}))

    def test_status_error(self, mock_lg: Logger) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "boom"
        err = httpx.HTTPStatusError("boom", request=MagicMock(), response=mock_resp)
        backend = self._setup_async(mock_lg, err)
        with pytest.raises(BackendRequestError):
            asyncio.run(backend._execute_async("http://x", {}))

    def test_request_error(self, mock_lg: Logger) -> None:
        backend = self._setup_async(mock_lg, httpx.RequestError("transport"))
        with pytest.raises(BackendRequestError):
            asyncio.run(backend._execute_async("http://x", {}))


# ---------------------------------------------------------------------------
# _parse_sse_line edge cases
# ---------------------------------------------------------------------------


class TestParseSseLine:
    def test_empty_line(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        assert backend._parse_sse_line("") is False
        backend.close()

    def test_non_data_line(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        assert backend._parse_sse_line("event: ping") is False
        backend.close()

    def test_done_marker(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        assert backend._parse_sse_line("data: [DONE]") is None
        backend.close()

    def test_data_no_space(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        result = backend._parse_sse_line('data:{"x": 1}')
        assert result == {"x": 1}
        backend.close()

    def test_data_with_space(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        result = backend._parse_sse_line('data: {"x": 1}')
        assert result == {"x": 1}
        backend.close()

    def test_invalid_json(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)
        with pytest.raises(BackendRequestError):
            backend._parse_sse_line("data: {invalid")
        backend.close()


# ---------------------------------------------------------------------------
# _StreamState
# ---------------------------------------------------------------------------


class TestStreamState:
    def test_empty_state_to_response(self) -> None:
        state = _StreamState()
        resp = state.to_response("model")
        assert resp.content == ""
        assert resp.thinking is None
        assert resp.tool_calls is None

    def test_content_chunks(self) -> None:
        state = _StreamState()
        token = state.process_chunk({"choices": [{"delta": {"content": "hello"}}]})
        assert token == "hello"
        token = state.process_chunk({"choices": [{"delta": {"content": " world"}}]})
        assert token == " world"
        resp = state.to_response("m")
        assert resp.content == "hello world"

    def test_thinking_chunks(self) -> None:
        state = _StreamState()
        token = state.process_chunk({"choices": [{"delta": {"thinking": "reasoning"}}]})
        assert token is None
        resp = state.to_response("m")
        assert resp.thinking == "reasoning"

    def test_finish_reason(self) -> None:
        state = _StreamState()
        state.process_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        assert state.finish_reason == FinishReason.STOP

    def test_usage_chunk(self) -> None:
        state = _StreamState()
        state.process_chunk({"usage": {"prompt_tokens": 5, "completion_tokens": 3}})
        assert state.usage is not None
        assert state.usage.prompt_tokens == 5

    def test_adapter_chunk(self) -> None:
        state = _StreamState()
        state.process_chunk(
            {"adapter": {"requested": "x", "actual": "x", "fallback": False}}
        )
        assert state.adapter is not None
        assert state.adapter.requested == "x"

    def test_no_choices(self) -> None:
        state = _StreamState()
        token = state.process_chunk({})
        assert token is None

    def test_tool_call_buffering(self) -> None:
        state = _StreamState()
        state.process_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc1",
                                    "function": {
                                        "name": "f",
                                        "arguments": "{",
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )
        # Continued chunk with more args
        state.process_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"x":1}'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        resp = state.to_response("m")
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].id == "tc1"
        assert resp.tool_calls[0].function.arguments == '{"x":1}'


# ---------------------------------------------------------------------------
# chat_stream_async
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Fake httpx.Response supporting async stream iteration."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCM:
    """Async context manager wrapping a fake stream response."""

    def __init__(
        self, lines: list[str] | None = None, raise_exc: Exception | None = None
    ) -> None:
        self._lines = lines or []
        self._raise = raise_exc

    async def __aenter__(self) -> Any:
        if self._raise is not None:
            raise self._raise
        return _FakeStreamResponse(self._lines)

    async def __aexit__(self, *args: Any) -> None:
        pass


def test_chat_stream_async(mock_lg: Logger) -> None:
    backend = _backend(mock_lg)
    async_client = backend._get_async_client()
    lines = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}',
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    async_client.stream = MagicMock(return_value=_FakeStreamCM(lines))

    request = ChatRequest(messages=[{"role": "user", "content": "hi"}])

    async def collect() -> list[str]:
        return [t async for t in backend.chat_stream_async(request)]

    tokens = asyncio.run(collect())
    assert tokens == ["hi", " there"]


def test_chat_stream_async_connection_error(mock_lg: Logger) -> None:
    backend = _backend(mock_lg)
    async_client = backend._get_async_client()
    async_client.stream = MagicMock(
        return_value=_FakeStreamCM(raise_exc=httpx.ConnectError("boom"))
    )

    request = ChatRequest(messages=[{"role": "user", "content": "hi"}])

    async def collect() -> list[str]:
        return [t async for t in backend.chat_stream_async(request)]

    with pytest.raises(BackendUnavailableError):
        asyncio.run(collect())


# ---------------------------------------------------------------------------
# _execute_stream_sync error path
# ---------------------------------------------------------------------------


class TestExecuteStreamSyncErrors:
    def test_connection_error(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)

        class _CM:
            def __enter__(self) -> Any:
                raise httpx.ConnectError("boom")

            def __exit__(self, *args: Any) -> None:
                pass

        backend._client.stream = MagicMock(return_value=_CM())
        with pytest.raises(BackendUnavailableError):
            list(backend._execute_stream_sync("http://x", {}))
        backend.close()

    def test_timeout(self, mock_lg: Logger) -> None:
        backend = _backend(mock_lg)

        class _CM:
            def __enter__(self) -> Any:
                raise httpx.TimeoutException("boom")

            def __exit__(self, *args: Any) -> None:
                pass

        backend._client.stream = MagicMock(return_value=_CM())
        with pytest.raises(BackendTimeoutError):
            list(backend._execute_stream_sync("http://x", {}))
        backend.close()
