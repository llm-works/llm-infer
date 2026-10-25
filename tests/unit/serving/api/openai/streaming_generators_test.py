# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for serving/api/openai/streaming_generators.py."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger
from appinfra.service.errors import ChannelError, ChannelTimeoutError

from llm_infer.response.parsers.think import ThinkTagNormalizer
from llm_infer.schemas.openai import FinishReason
from llm_infer.serving.api.openai.streaming_generators import (
    ChatStreamingGenerator,
    CompletionStreamingGenerator,
    _extract_adapter_info,
    _map_finish_reason,
)
from llm_infer.serving.dispatch.types import (
    Request as InternalRequest,
)
from llm_infer.serving.dispatch.types import (
    ResponseAdapterInfo,
    StreamChunk,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _map_finish_reason
# ---------------------------------------------------------------------------


class TestMapFinishReason:
    def test_passes_reason_through(self) -> None:
        assert _map_finish_reason(FinishReason.LENGTH) == FinishReason.LENGTH
        assert _map_finish_reason(FinishReason.TOOL_CALLS) == FinishReason.TOOL_CALLS

    def test_none_reports_stop(self) -> None:
        assert _map_finish_reason(None) == FinishReason.STOP


# ---------------------------------------------------------------------------
# _extract_adapter_info
# ---------------------------------------------------------------------------


class TestExtractAdapterInfo:
    def test_no_adapter(self) -> None:
        chunk = MagicMock(spec=[])
        assert _extract_adapter_info(chunk) is None

    def test_adapter_none(self) -> None:
        chunk = MagicMock()
        chunk.adapter = None
        assert _extract_adapter_info(chunk) is None

    def test_adapter_present(self) -> None:
        chunk = MagicMock()
        chunk.adapter = ResponseAdapterInfo(
            requested="x", actual="x", fallback=False, mtime="t", md5="m"
        )
        result = _extract_adapter_info(chunk)
        assert result is not None
        assert result.requested == "x"


# ---------------------------------------------------------------------------
# Stub IPC for streaming
# ---------------------------------------------------------------------------


class _StubIPC:
    def __init__(
        self,
        chunks: list[StreamChunk] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.raise_exc = raise_exc

    def submit_stream(self, request: Any) -> Any:
        chunks = self.chunks
        raise_exc = self.raise_exc

        async def _gen() -> Any:
            if raise_exc is not None:
                raise raise_exc
            for c in chunks:
                yield c

        return _gen()


def _request() -> InternalRequest:
    return InternalRequest(id="r1", prompt="hi", max_tokens=100)


async def _collect(generator: Any) -> list[str]:
    return [chunk async for chunk in generator]


def _final_finish_reason(events: list[str]) -> str:
    """Return finish_reason from the last data event before [DONE]."""
    payload = json.loads(events[-2].removeprefix("data: ").strip())
    reason: str = payload["choices"][0]["finish_reason"]
    return reason


# ---------------------------------------------------------------------------
# ChatStreamingGenerator
# ---------------------------------------------------------------------------


class TestChatStreamingGenerator:
    def _make(
        self,
        chunks: list[StreamChunk] | None = None,
        raise_exc: Exception | None = None,
        normalizer: ThinkTagNormalizer | None = None,
        max_tokens: int | None = None,
    ) -> ChatStreamingGenerator:
        ipc = _StubIPC(chunks=chunks, raise_exc=raise_exc)
        lg = MagicMock(spec=Logger)
        return ChatStreamingGenerator(lg, "r1", "model", ipc, normalizer, max_tokens)

    def test_basic_stream(self) -> None:
        chunks = [
            StreamChunk(id="r1", token="hello"),
            StreamChunk(id="r1", token=" world"),
            StreamChunk(
                id="r1",
                token="",
                is_final=True,
                finish_reason="stop",
                prompt_tokens=2,
                completion_tokens=2,
            ),
        ]
        gen = self._make(chunks=chunks)
        result = asyncio.run(_collect(gen.stream(_request())))
        # role + 2 content + final + [DONE]
        assert len(result) == 5
        assert "[DONE]" in result[-1]
        assert "assistant" in result[0]

    def test_normalizer_buffers(self) -> None:
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        chunks = [
            StreamChunk(id="r1", token="<th"),  # Buffered (too short)
            StreamChunk(id="r1", token="inking>r</thinking>plenty more text"),
            StreamChunk(id="r1", token="", is_final=True, finish_reason="stop"),
        ]
        gen = self._make(chunks=chunks, normalizer=normalizer)
        result = asyncio.run(_collect(gen.stream(_request())))
        # First content chunk was buffered
        # Subsequent chunks should appear after normalization
        full = "".join(result)
        assert "<think>" in full or "</think>" in full

    def test_timeout_emits_error_event(self) -> None:
        gen = self._make(raise_exc=TimeoutError("ipc timeout"))
        result = asyncio.run(_collect(gen.stream(_request())))
        # Header chunk + error event + [DONE]
        assert "timeout" in result[-2]
        assert "[DONE]" in result[-1]

    def test_channel_timeout_emits_timeout_event(self) -> None:
        gen = self._make(raise_exc=ChannelTimeoutError("no chunk in 30s"))
        result = asyncio.run(_collect(gen.stream(_request())))
        error = json.loads(result[-2].removeprefix("data: ").strip())["error"]
        assert error["code"] == "timeout"
        assert "[DONE]" in result[-1]

    def test_failed_generation_emits_error_event(self) -> None:
        """A chunk with ``error`` set makes the channel raise ChannelError."""
        gen = self._make(raise_exc=ChannelError("Request failed: bad adapter"))
        result = asyncio.run(_collect(gen.stream(_request())))
        error = json.loads(result[-2].removeprefix("data: ").strip())["error"]
        assert error["code"] == "error"
        assert "bad adapter" in error["message"]
        assert "[DONE]" in result[-1]

    def test_max_tokens_overrides_finish_reason(self) -> None:
        chunks = [
            StreamChunk(id="r1", token="hello"),
            StreamChunk(
                id="r1",
                token="",
                is_final=True,
                finish_reason="stop",
                completion_tokens=100,
            ),
        ]
        gen = self._make(chunks=chunks, max_tokens=100)
        result = asyncio.run(_collect(gen.stream(_request())))
        # Final chunk should have finish_reason=length
        # Find the chunk with finish_reason
        full = "".join(result)
        assert "length" in full

    def _tool_call_final(self, completion_tokens: int) -> list[StreamChunk]:
        return [
            StreamChunk(
                id="r1",
                token="",
                is_final=True,
                finish_reason="tool_calls",
                tool_calls=[
                    {"function": {"name": "f", "arguments": '{"a": '}, "id": "tc1"}
                ],
                completion_tokens=completion_tokens,
            ),
        ]

    def test_tool_calls_finish_reason_preserved_under_limit(self) -> None:
        gen = self._make(chunks=self._tool_call_final(50), max_tokens=100)
        result = asyncio.run(_collect(gen.stream(_request())))
        assert _final_finish_reason(result) == "tool_calls"

    def test_tool_calls_at_limit_reports_length(self) -> None:
        """Truncated tool-call arguments must not look like a complete call."""
        gen = self._make(chunks=self._tool_call_final(100), max_tokens=100)
        result = asyncio.run(_collect(gen.stream(_request())))
        assert _final_finish_reason(result) == "length"

    def test_empty_token_chunks_ignored(self) -> None:
        chunks = [
            StreamChunk(id="r1", token=""),  # Empty token, skipped
            StreamChunk(id="r1", token="hello"),
            StreamChunk(id="r1", token="", is_final=True, finish_reason="stop"),
        ]
        gen = self._make(chunks=chunks)
        result = asyncio.run(_collect(gen.stream(_request())))
        # role + 1 content (empty token skipped) + final + [DONE]
        assert len(result) == 4

    def test_create_content_chunk_buffered_returns_empty(self) -> None:
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        gen = self._make(normalizer=normalizer)
        # Short input gets buffered
        result = gen.create_content_chunk("<th")
        assert result == ""

    def test_create_final_chunk_with_normalizer_flush(self) -> None:
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        normalizer.process("<thi")  # Leave content in buffer
        gen = self._make(normalizer=normalizer)
        result = gen.create_final_chunk(FinishReason.STOP)
        # Should contain the flushed content somewhere
        assert "data:" in result


# ---------------------------------------------------------------------------
# CompletionStreamingGenerator
# ---------------------------------------------------------------------------


class TestCompletionStreamingGenerator:
    def _make(
        self,
        chunks: list[StreamChunk] | None = None,
        raise_exc: Exception | None = None,
    ) -> CompletionStreamingGenerator:
        ipc = _StubIPC(chunks=chunks, raise_exc=raise_exc)
        lg = MagicMock(spec=Logger)
        return CompletionStreamingGenerator(lg, "r1", "model", ipc)

    def test_basic_stream(self) -> None:
        chunks = [
            StreamChunk(id="r1", token="hello"),
            StreamChunk(id="r1", token="", is_final=True, finish_reason="stop"),
        ]
        gen = self._make(chunks=chunks)
        result = asyncio.run(_collect(gen.stream(_request())))
        # No header for legacy completion - 1 content + final + [DONE]
        assert len(result) == 3

    def test_no_header(self) -> None:
        """Legacy completions don't emit a header chunk."""
        gen = self._make(chunks=[])
        assert gen.create_header_chunk() is None

    def test_create_content_chunk(self) -> None:
        gen = self._make()
        result = gen.create_content_chunk("hello")
        assert "data:" in result
        assert "hello" in result

    def test_create_final_chunk(self) -> None:
        gen = self._make()
        result = gen.create_final_chunk(FinishReason.STOP)
        assert "data:" in result

    def test_timeout(self) -> None:
        gen = self._make(raise_exc=TimeoutError("timeout"))
        result = asyncio.run(_collect(gen.stream(_request())))
        assert "timeout" in result[-2]
        assert "[DONE]" in result[-1]
