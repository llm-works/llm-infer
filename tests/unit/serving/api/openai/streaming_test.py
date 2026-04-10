"""Unit tests for serving/api/openai/streaming.py."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from llm_infer.schemas.openai import (
    AdapterInfoResponse,
    ChatCompletionChunk,
    CompletionChunk,
    FinishReason,
    Role,
)
from llm_infer.serving.api.openai.streaming import (
    _convert_tool_calls_to_deltas,
    create_chat_chunk,
    create_completion_chunk,
    format_sse_done,
    format_sse_error,
    format_sse_event,
    stream_chat_completion,
    stream_chat_completion_sync,
    stream_completion_sync,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# format_sse_*
# ---------------------------------------------------------------------------


def test_format_sse_event() -> None:
    assert format_sse_event("hello") == "data: hello\n\n"


def test_format_sse_done() -> None:
    assert format_sse_done() == "data: [DONE]\n\n"


def test_format_sse_error_default() -> None:
    out = format_sse_error("boom")
    assert out.startswith("data: ")
    payload = json.loads(out[len("data: ") :].strip())
    assert payload["error"]["message"] == "boom"
    assert payload["error"]["type"] == "server_error"
    assert payload["error"]["code"] == "error"


def test_format_sse_error_custom() -> None:
    out = format_sse_error("timed out", error_type="timeout_error", code="timeout")
    payload = json.loads(out[len("data: ") :].strip())
    assert payload["error"]["type"] == "timeout_error"
    assert payload["error"]["code"] == "timeout"


# ---------------------------------------------------------------------------
# _convert_tool_calls_to_deltas
# ---------------------------------------------------------------------------


class TestConvertToolCallsToDeltas:
    def test_none(self) -> None:
        assert _convert_tool_calls_to_deltas(None) is None

    def test_empty(self) -> None:
        assert _convert_tool_calls_to_deltas([]) is None

    def test_single_tool_call(self) -> None:
        result = _convert_tool_calls_to_deltas(
            [{"function": {"name": "f", "arguments": "{}"}, "id": "abc"}]
        )
        assert result is not None
        assert len(result) == 1
        assert result[0].id == "abc"
        assert result[0].function.name == "f"

    def test_skip_malformed_no_name(self) -> None:
        result = _convert_tool_calls_to_deltas([{"function": {}, "id": "abc"}])
        assert result is None

    def test_indices_are_sequential_after_skip(self) -> None:
        result = _convert_tool_calls_to_deltas(
            [
                {"function": {}, "id": "skip"},  # malformed
                {"function": {"name": "f1", "arguments": "{}"}, "id": "a"},
                {"function": {"name": "f2", "arguments": "{}"}, "id": "b"},
            ]
        )
        assert result is not None
        assert len(result) == 2
        assert result[0].index == 0
        assert result[1].index == 1


# ---------------------------------------------------------------------------
# create_chat_chunk / create_completion_chunk
# ---------------------------------------------------------------------------


class TestCreateChunks:
    def test_chat_chunk_minimal(self) -> None:
        chunk = create_chat_chunk("r1", "model", 1234567890)
        assert isinstance(chunk, ChatCompletionChunk)
        assert chunk.id == "r1"
        assert chunk.model == "model"
        assert chunk.created == 1234567890

    def test_chat_chunk_with_role(self) -> None:
        chunk = create_chat_chunk("r1", "m", 0, role=Role.ASSISTANT)
        assert chunk.choices[0].delta.role == Role.ASSISTANT

    def test_chat_chunk_with_content(self) -> None:
        chunk = create_chat_chunk("r1", "m", 0, content="hello")
        assert chunk.choices[0].delta.content == "hello"

    def test_chat_chunk_with_finish_reason(self) -> None:
        chunk = create_chat_chunk("r1", "m", 0, finish_reason=FinishReason.STOP)
        assert chunk.choices[0].finish_reason == FinishReason.STOP

    def test_chat_chunk_with_tool_calls(self) -> None:
        chunk = create_chat_chunk(
            "r1",
            "m",
            0,
            tool_calls=[{"function": {"name": "f", "arguments": "{}"}, "id": "tc1"}],
        )
        assert chunk.choices[0].delta.tool_calls is not None

    def test_chat_chunk_with_adapter(self) -> None:
        adapter = AdapterInfoResponse(requested="x", actual="x", fallback=False)
        chunk = create_chat_chunk("r1", "m", 0, adapter=adapter)
        assert chunk.adapter is not None

    def test_completion_chunk_minimal(self) -> None:
        chunk = create_completion_chunk("r1", "m", 0, text="hello")
        assert isinstance(chunk, CompletionChunk)
        assert chunk.choices[0].text == "hello"

    def test_completion_chunk_with_finish_reason(self) -> None:
        chunk = create_completion_chunk(
            "r1", "m", 0, text="", finish_reason=FinishReason.STOP
        )
        assert chunk.choices[0].finish_reason == FinishReason.STOP


# ---------------------------------------------------------------------------
# stream_chat_completion (async)
# ---------------------------------------------------------------------------


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_stream_chat_completion_full() -> None:
    chunks = []
    async for chunk in stream_chat_completion(
        "r1",
        "model",
        _async_iter(["hi", " there"]),
        get_finish_reason=lambda: FinishReason.STOP,
    ):
        chunks.append(chunk)
    # First chunk: role
    # Then 2 content chunks
    # Then final chunk with finish_reason
    # Then [DONE]
    assert len(chunks) == 5
    assert "[DONE]" in chunks[-1]
    # First chunk has assistant role
    assert "assistant" in chunks[0]


@pytest.mark.asyncio
async def test_stream_chat_completion_no_tokens() -> None:
    chunks = []
    async for chunk in stream_chat_completion(
        "r1", "model", _async_iter([]), get_finish_reason=lambda: FinishReason.STOP
    ):
        chunks.append(chunk)
    # role + final + [DONE]
    assert len(chunks) == 3


# ---------------------------------------------------------------------------
# stream_chat_completion_sync
# ---------------------------------------------------------------------------


def test_stream_chat_completion_sync() -> None:
    chunks = list(
        stream_chat_completion_sync(
            "r1",
            "model",
            iter(["hi"]),
            get_finish_reason=lambda: FinishReason.STOP,
        )
    )
    # role + 1 content + final + [DONE]
    assert len(chunks) == 4
    assert "[DONE]" in chunks[-1]


def test_stream_chat_completion_sync_with_adapter() -> None:
    adapter = AdapterInfoResponse(requested="x", actual="x", fallback=False)
    chunks = list(
        stream_chat_completion_sync(
            "r1",
            "model",
            iter([]),
            get_finish_reason=lambda: FinishReason.STOP,
            adapter=adapter,
        )
    )
    # Adapter info appears in the final chunk
    final = chunks[-2]  # second-to-last is the final chunk; last is [DONE]
    assert "adapter" in final


# ---------------------------------------------------------------------------
# stream_completion_sync (legacy)
# ---------------------------------------------------------------------------


def test_stream_completion_sync() -> None:
    chunks = list(
        stream_completion_sync(
            "r1",
            "model",
            iter(["hello"]),
            get_finish_reason=lambda: FinishReason.STOP,
        )
    )
    # 1 content + final + [DONE]
    assert len(chunks) == 3
    assert "[DONE]" in chunks[-1]


def test_stream_completion_sync_empty() -> None:
    chunks = list(
        stream_completion_sync("r1", "model", iter([]), get_finish_reason=lambda: None)
    )
    # final + [DONE]
    assert len(chunks) == 2
