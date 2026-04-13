"""Additional Anthropic backend tests covering chat methods, streaming, errors.

Existing test_anthropic_backend.py covers _convert_messages, _prepare_request,
structured output, and from_config. This file fills in:
- chat() / chat_async() / chat_stream() / chat_stream_async() success paths
- _handle_errors exception translation
- _process_stream_event branches
- _StreamState
- _parse_response variants
- close() / aclose()
- last_response / think NotImplementedError
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from appinfra.log import Logger

from llm_infer.client.errors import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_infer.schemas.openai import FinishReason

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    return MagicMock(spec=Logger)


@pytest.fixture
def mock_anthropic() -> Any:
    """Create mock anthropic module with Anthropic and AsyncAnthropic."""
    mock_module = MagicMock()
    mock_module.Anthropic.return_value = MagicMock()
    mock_module.AsyncAnthropic.return_value = MagicMock()
    mock_module.APIConnectionError = type(
        "APIConnectionError", (Exception,), {"message": "conn"}
    )
    mock_module.APITimeoutError = type(
        "APITimeoutError", (Exception,), {"message": "timeout"}
    )
    mock_module.APIStatusError = type(
        "APIStatusError",
        (Exception,),
        {"message": "boom", "status_code": 500},
    )
    return mock_module


def _make_backend(mock_anthropic: Any, mock_lg: Logger) -> Any:
    """Create AnthropicBackend with mocked anthropic SDK."""
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        from llm_infer.client.backends.anthropic import AnthropicBackend

        return AnthropicBackend(lg=mock_lg, api_key="test-key")


def _content_block(block_type: str, **fields: Any) -> Any:
    """Create a mock content block."""
    block = MagicMock()
    block.type = block_type
    for k, v in fields.items():
        setattr(block, k, v)
    return block


def _mock_response(
    *,
    content: list[Any] | None = None,
    stop_reason: str | None = "end_turn",
    usage_input: int = 5,
    usage_output: int = 3,
    model: str = "claude-test",
) -> Any:
    response = MagicMock()
    response.content = content or [_content_block("text", text="hello")]
    response.stop_reason = stop_reason
    response.model = model
    usage = MagicMock()
    usage.input_tokens = usage_input
    usage.output_tokens = usage_output
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# last_response property
# ---------------------------------------------------------------------------


def test_last_response_initially_none(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    assert backend.last_response is None


# ---------------------------------------------------------------------------
# chat() sync
# ---------------------------------------------------------------------------


def test_chat_success(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    backend._client.messages.create.return_value = _mock_response()
    result = backend.chat([{"role": "user", "content": "hi"}])
    assert result.content == "hello"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3
    assert backend.last_response is result


def test_chat_with_tool_calls(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    tool_block = _content_block("tool_use", id="tc1", name="my_func", input={"a": 1})
    backend._client.messages.create.return_value = _mock_response(
        content=[tool_block], stop_reason="tool_use"
    )
    result = backend.chat([{"role": "user", "content": "hi"}])
    assert result.tool_calls is not None
    assert result.tool_calls[0].function.name == "my_func"
    assert result.finish_reason == FinishReason.TOOL_CALLS


def test_chat_with_thinking(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    blocks = [
        _content_block("thinking", thinking="reasoning"),
        _content_block("text", text="answer"),
    ]
    backend._client.messages.create.return_value = _mock_response(content=blocks)
    result = backend.chat([{"role": "user", "content": "hi"}])
    assert result.content == "answer"
    assert result.thinking == "reasoning"


def test_chat_think_mode_raises(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    with pytest.raises(NotImplementedError, match="think mode"):
        backend.chat([{"role": "user", "content": "hi"}], think=True)


def test_chat_adapter_param_ignored(mock_anthropic: Any, mock_lg: Logger) -> None:
    """Adapter param is silently ignored for Anthropic."""
    backend = _make_backend(mock_anthropic, mock_lg)
    backend._client.messages.create.return_value = _mock_response()
    # Should not raise even though Anthropic doesn't support adapters
    result = backend.chat([{"role": "user", "content": "hi"}], adapter="my-adapter")
    assert result.content == "hello"


def test_chat_propagates_connection_error(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    backend._client.messages.create.side_effect = mock_anthropic.APIConnectionError(
        "boom"
    )
    with pytest.raises(BackendUnavailableError):
        backend.chat([{"role": "user", "content": "hi"}])


def test_chat_propagates_timeout_error(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    backend._client.messages.create.side_effect = mock_anthropic.APITimeoutError(
        "timeout"
    )
    with pytest.raises(BackendTimeoutError):
        backend.chat([{"role": "user", "content": "hi"}])


def test_chat_propagates_status_error(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    err = mock_anthropic.APIStatusError("boom")
    err.message = "boom"
    err.status_code = 503
    backend._client.messages.create.side_effect = err
    with pytest.raises(BackendRequestError) as exc_info:
        backend.chat([{"role": "user", "content": "hi"}])
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# chat_async()
# ---------------------------------------------------------------------------


def test_chat_async_success(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)

    async def fake_create(**kwargs: Any) -> Any:
        return _mock_response()

    async_client = mock_anthropic.AsyncAnthropic.return_value
    async_client.messages.create = fake_create

    result = asyncio.run(backend.chat_async([{"role": "user", "content": "hi"}]))
    assert result.content == "hello"


def test_chat_async_propagates_connection_error(
    mock_anthropic: Any, mock_lg: Logger
) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)

    async def fake_create(**kwargs: Any) -> Any:
        raise mock_anthropic.APIConnectionError("boom")

    async_client = mock_anthropic.AsyncAnthropic.return_value
    async_client.messages.create = fake_create

    with pytest.raises(BackendUnavailableError):
        asyncio.run(backend.chat_async([{"role": "user", "content": "hi"}]))


def test_chat_async_propagates_timeout(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)

    async def fake_create(**kwargs: Any) -> Any:
        raise mock_anthropic.APITimeoutError("timeout")

    async_client = mock_anthropic.AsyncAnthropic.return_value
    async_client.messages.create = fake_create

    with pytest.raises(BackendTimeoutError):
        asyncio.run(backend.chat_async([{"role": "user", "content": "hi"}]))


def test_chat_async_propagates_status_error(
    mock_anthropic: Any, mock_lg: Logger
) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)

    async def fake_create(**kwargs: Any) -> Any:
        err = mock_anthropic.APIStatusError("boom")
        err.message = "bad"
        err.status_code = 500
        raise err

    async_client = mock_anthropic.AsyncAnthropic.return_value
    async_client.messages.create = fake_create

    with pytest.raises(BackendRequestError):
        asyncio.run(backend.chat_async([{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------
# Streaming - sync
# ---------------------------------------------------------------------------


class _FakeSyncStream:
    """Mock sync stream supporting the context manager + iteration protocol."""

    def __init__(self, events: list[Any], final_message: Any) -> None:
        self._events = events
        self._final_message = final_message

    def __enter__(self) -> _FakeSyncStream:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def __iter__(self) -> Any:
        return iter(self._events)

    def get_final_message(self) -> Any:
        return self._final_message


def _delta_event(text: str) -> Any:
    event = MagicMock()
    event.type = "content_block_delta"
    delta = MagicMock()
    delta.type = "text_delta"
    delta.text = text
    event.delta = delta
    return event


def test_chat_stream_yields_text(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    events = [_delta_event("hello "), _delta_event("world")]
    final = _mock_response(stop_reason="end_turn")
    backend._client.messages.stream.return_value = _FakeSyncStream(events, final)

    tokens = list(backend.chat_stream([{"role": "user", "content": "hi"}]))
    assert tokens == ["hello ", "world"]
    assert backend.last_response is not None
    assert backend.last_response.content == "hello world"


# ---------------------------------------------------------------------------
# Streaming - async
# ---------------------------------------------------------------------------


class _FakeAsyncStream:
    def __init__(self, events: list[Any], final_message: Any) -> None:
        self._events = events
        self._final_message = final_message

    async def __aenter__(self) -> _FakeAsyncStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __aiter__(self) -> _FakeAsyncStream:
        self._idx = 0
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._idx]
        self._idx += 1
        return event

    async def get_final_message(self) -> Any:
        return self._final_message


def test_chat_stream_async_yields_text(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    events = [_delta_event("hi"), _delta_event(" there")]
    final = _mock_response(stop_reason="end_turn")
    async_client = mock_anthropic.AsyncAnthropic.return_value
    async_client.messages.stream.return_value = _FakeAsyncStream(events, final)

    async def collect() -> list[str]:
        return [
            t
            async for t in backend.chat_stream_async(
                [{"role": "user", "content": "hi"}]
            )
        ]

    tokens = asyncio.run(collect())
    assert tokens == ["hi", " there"]


# ---------------------------------------------------------------------------
# _process_stream_event branches
# ---------------------------------------------------------------------------


def test_process_stream_event_text_delta(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    state = _StreamState()
    token = backend._process_stream_event(_delta_event("hi"), state)
    assert token == "hi"
    assert state.content_parts == ["hi"]


def test_process_stream_event_thinking_delta(
    mock_anthropic: Any, mock_lg: Logger
) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    event = MagicMock()
    event.type = "content_block_delta"
    delta = MagicMock()
    delta.type = "thinking_delta"
    delta.thinking = "reasoning"
    event.delta = delta

    state = _StreamState()
    token = backend._process_stream_event(event, state)
    assert token is None
    assert state.thinking_parts == ["reasoning"]


def test_process_stream_event_no_delta(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = None

    state = _StreamState()
    assert backend._process_stream_event(event, state) is None


def test_process_stream_event_block_stop_tool_use(
    mock_anthropic: Any, mock_lg: Logger
) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    event = MagicMock()
    event.type = "content_block_stop"
    block = MagicMock()
    block.type = "tool_use"
    block.id = "tc1"
    block.name = "my_func"
    block.input = {"x": 1}
    event.content_block = block

    state = _StreamState()
    backend._process_stream_event(event, state)
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].function.name == "my_func"


def test_process_stream_event_message_delta_usage(
    mock_anthropic: Any, mock_lg: Logger
) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    event = MagicMock()
    event.type = "message_delta"
    usage = MagicMock()
    usage.input_tokens = 5
    usage.output_tokens = 3
    event.usage = usage

    state = _StreamState()
    backend._process_stream_event(event, state)
    assert state.usage is not None
    assert state.usage.prompt_tokens == 5


def test_process_stream_event_message_delta_no_usage(
    mock_anthropic: Any, mock_lg: Logger
) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    event = MagicMock()
    event.type = "message_delta"
    event.usage = None

    state = _StreamState()
    backend._process_stream_event(event, state)
    assert state.usage is None


# ---------------------------------------------------------------------------
# _create_usage
# ---------------------------------------------------------------------------


def test_create_usage_none(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    assert backend._create_usage(None) is None


def test_create_usage_present(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    result = backend._create_usage(usage)
    assert result is not None
    assert result.prompt_tokens == 10
    assert result.total_tokens == 15


# ---------------------------------------------------------------------------
# _create_tool_call
# ---------------------------------------------------------------------------


def test_create_tool_call_dict_input(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    block = MagicMock()
    block.id = "tc1"
    block.name = "f"
    block.input = {"arg": 1}
    tc = backend._create_tool_call(block)
    assert tc.id == "tc1"
    assert tc.function.name == "f"
    # Dict input gets serialized to JSON string
    assert json.loads(tc.function.arguments) == {"arg": 1}


def test_create_tool_call_string_input(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    block = MagicMock()
    block.id = "tc1"
    block.name = "f"
    block.input = '{"already": "json"}'
    tc = backend._create_tool_call(block)
    assert tc.function.arguments == '{"already": "json"}'


# ---------------------------------------------------------------------------
# _StreamState.to_response
# ---------------------------------------------------------------------------


def test_stream_state_to_response_basic(mock_anthropic: Any, mock_lg: Logger) -> None:
    _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    state = _StreamState()
    state.content_parts = ["hello ", "world"]
    state.thinking_parts = ["reasoning"]
    state.finish_reason = FinishReason.STOP

    resp = state.to_response("test-model")
    assert resp.content == "hello world"
    assert resp.thinking == "reasoning"
    assert resp.model == "test-model"


def test_stream_state_to_response_structured_output_overrides_finish(
    mock_anthropic: Any, mock_lg: Logger
) -> None:
    _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    state = _StreamState(structured_output_tool="__structured_output__")
    state.finish_reason = FinishReason.TOOL_CALLS
    resp = state.to_response("model")
    # Structured output overrides TOOL_CALLS -> STOP
    assert resp.finish_reason == FinishReason.STOP


def test_stream_state_to_response_no_thinking(
    mock_anthropic: Any, mock_lg: Logger
) -> None:
    _make_backend(mock_anthropic, mock_lg)
    from llm_infer.client.backends.anthropic import _StreamState

    state = _StreamState()
    state.content_parts = ["text"]
    resp = state.to_response("model")
    assert resp.thinking is None


# ---------------------------------------------------------------------------
# close() / aclose() / _get_async_client
# ---------------------------------------------------------------------------


def test_close_calls_client_close(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    backend.close()
    backend._client.close.assert_called_once()


def test_aclose_no_async_client(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    asyncio.run(backend.aclose())
    backend._client.close.assert_called_once()


def test_aclose_with_async_client(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    # Trigger async client creation
    async_client = backend._get_async_client()

    async def fake_close() -> None:
        return None

    async_client.close = fake_close
    asyncio.run(backend.aclose())
    backend._client.close.assert_called_once()
    assert backend._async_client is None  # Reset after close


def test_get_async_client_lazy_init(mock_anthropic: Any, mock_lg: Logger) -> None:
    backend = _make_backend(mock_anthropic, mock_lg)
    assert backend._async_client is None
    client = backend._get_async_client()
    assert client is not None
    # Subsequent call returns same instance
    assert backend._get_async_client() is client
