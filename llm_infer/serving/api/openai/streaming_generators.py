"""Streaming response generators using Template Method pattern."""

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
    from ...dispatch.types import Request as InternalRequest


def _map_finish_reason(reason: str | None) -> FinishReason:
    """Map internal finish reason to OpenAI finish reason."""
    if reason == "length":
        return FinishReason.LENGTH
    if reason == "error":
        return FinishReason.CONTENT_FILTER
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
        """Create SSE event for a content token."""
        pass

    @abstractmethod
    def create_final_chunk(self, finish_reason: FinishReason) -> str:
        """Create SSE event for the final chunk with finish reason."""
        pass

    async def stream(self, internal_request: "InternalRequest") -> AsyncIterator[str]:
        """Template method: execute the streaming algorithm."""
        # Optional header chunk
        header = self.create_header_chunk()
        if header:
            yield header

        # Stream tokens
        finish_reason = FinishReason.STOP
        async for chunk in self.ipc.submit_streaming(self.request_id, internal_request):
            if chunk.is_final:
                finish_reason = _map_finish_reason(chunk.finish_reason)
                break
            if chunk.token:
                yield self.create_content_chunk(chunk.token)

        # Final chunk with finish_reason
        yield self.create_final_chunk(finish_reason)
        yield format_sse_done()


class ChatStreamingGenerator(StreamingGenerator):
    """Streaming generator for chat completions."""

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
        chunk = create_chat_chunk(
            request_id=self.request_id,
            model=self.model,
            created=self.created,
            content=token,
        )
        return format_sse_event(chunk.model_dump_json())

    def create_final_chunk(self, finish_reason: FinishReason) -> str:
        """Create final chunk for chat."""
        chunk = create_chat_chunk(
            request_id=self.request_id,
            model=self.model,
            created=self.created,
            finish_reason=finish_reason,
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

    def create_final_chunk(self, finish_reason: FinishReason) -> str:
        """Create final chunk for completion."""
        chunk = create_completion_chunk(
            request_id=self.request_id,
            model=self.model,
            created=self.created,
            text="",
            finish_reason=finish_reason,
        )
        return format_sse_event(chunk.model_dump_json())
