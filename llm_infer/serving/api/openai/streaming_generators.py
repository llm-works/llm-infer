"""Streaming response generators using Template Method pattern."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from llm_infer.schemas.openai import FinishReason, Role

from .streaming import (
    create_chat_chunk,
    create_completion_chunk,
    format_sse_done,
    format_sse_event,
)

if TYPE_CHECKING:
    from llm_infer.response.parsers.think import ThinkTagNormalizer

    from ...dispatch.types import Request as InternalRequest


def _map_finish_reason(reason: str | None) -> FinishReason:
    """Map internal finish reason to OpenAI finish reason.

    Note: Internal "error" maps to STOP since OpenAI's FinishReason enum
    doesn't include an ERROR value. Error details are surfaced separately
    via HTTP status codes and error responses for non-streaming requests.
    """
    if reason == "length":
        return FinishReason.LENGTH
    if reason == "tool_calls":
        return FinishReason.TOOL_CALLS
    # "error" and all other cases map to STOP
    return FinishReason.STOP


class StreamingGenerator(ABC):
    """Abstract base for SSE streaming generators.

    Template Method pattern: defines the streaming algorithm skeleton,
    with subclasses providing chunk creation specifics.
    """

    def __init__(self, request_id: str, model: str, ipc: Any):
        self.request_id = request_id
        self.model = model
        self.ipc = ipc
        self.created = int(time.time())

    def create_header_chunk(self) -> str | None:
        """Create optional header chunk before streaming tokens.

        Override in subclasses that need a header (e.g., chat role announcement).
        Returns None if no header is needed.
        """
        return None

    @abstractmethod
    def create_content_chunk(self, token: str) -> str:
        """Create SSE event for a content token.

        Returns SSE-formatted string, or empty string if content is buffered
        (e.g., by a normalizer waiting for complete tags).
        """
        pass

    @abstractmethod
    def create_final_chunk(
        self,
        finish_reason: FinishReason,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create SSE event for the final chunk with finish reason."""
        pass

    async def stream(self, internal_request: InternalRequest) -> AsyncIterator[str]:
        """Template method: execute the streaming algorithm."""
        # Optional header chunk
        header = self.create_header_chunk()
        if header:
            yield header

        # Stream tokens
        finish_reason = FinishReason.STOP
        tool_calls = None
        async for chunk in self.ipc.submit_streaming(self.request_id, internal_request):
            if chunk.is_final:
                finish_reason = _map_finish_reason(chunk.finish_reason)
                tool_calls = getattr(chunk, "tool_calls", None)
                break
            if chunk.token:
                content = self.create_content_chunk(chunk.token)
                if content:  # Skip empty chunks (e.g., normalizer buffering)
                    yield content

        # Final chunk with finish_reason and tool_calls
        yield self.create_final_chunk(finish_reason, tool_calls)
        yield format_sse_done()


class ChatStreamingGenerator(StreamingGenerator):
    """Streaming generator for chat completions.

    Note: Think tags stream inline in content (not separated into thinking field).
    This matches vLLM behavior and allows clients to parse tags themselves.
    The normalizer only converts tag variants (e.g. <thinking> -> <think>).
    """

    def __init__(
        self,
        request_id: str,
        model: str,
        ipc: Any,
        normalizer: ThinkTagNormalizer | None = None,
    ):
        super().__init__(request_id, model, ipc)
        self._normalizer = normalizer

    def create_header_chunk(self) -> str:
        """Create role announcement chunk."""
        chunk = create_chat_chunk(
            request_id=self.request_id,
            model=self.model,
            created=self.created,
            role=Role.ASSISTANT,
        )
        return format_sse_event(chunk.model_dump_json())

    def create_content_chunk(self, token: str) -> str:
        """Create content chunk for chat."""
        if self._normalizer:
            token = self._normalizer.process(token)
            if not token:  # Buffered, nothing to emit yet
                return ""
        chunk = create_chat_chunk(
            request_id=self.request_id,
            model=self.model,
            created=self.created,
            content=token,
        )
        return format_sse_event(chunk.model_dump_json())

    def create_final_chunk(
        self,
        finish_reason: FinishReason,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create final chunk for chat."""
        # Flush normalizer buffer if active
        flushed = ""
        if self._normalizer:
            flushed = self._normalizer.flush()

        chunk = create_chat_chunk(
            request_id=self.request_id,
            model=self.model,
            created=self.created,
            content=flushed if flushed else None,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
        return format_sse_event(chunk.model_dump_json())


class CompletionStreamingGenerator(StreamingGenerator):
    """Streaming generator for legacy completions."""

    def create_content_chunk(self, token: str) -> str:
        """Create content chunk for completion."""
        chunk = create_completion_chunk(
            request_id=self.request_id,
            model=self.model,
            created=self.created,
            text=token,
        )
        return format_sse_event(chunk.model_dump_json())

    def create_final_chunk(
        self,
        finish_reason: FinishReason,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create final chunk for completion.

        Note: Legacy completions don't support tool calls, so tool_calls is ignored.
        """
        chunk = create_completion_chunk(
            request_id=self.request_id,
            model=self.model,
            created=self.created,
            text="",
            finish_reason=finish_reason,
        )
        return format_sse_event(chunk.model_dump_json())
