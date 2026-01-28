"""SSE streaming utilities for OpenAI-compatible responses."""

import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from llm_infer.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    CompletionChunk,
    CompletionChunkChoice,
    FinishReason,
    FunctionCall,
    Role,
    ToolCallDelta,
)


def format_sse_event(data: str) -> str:
    """Format data as SSE event."""
    return f"data: {data}\n\n"


def format_sse_done() -> str:
    """Format SSE done marker."""
    return "data: [DONE]\n\n"


def _convert_tool_calls_to_deltas(
    tool_calls: list[dict[str, Any]] | None,
) -> list[ToolCallDelta] | None:
    """Convert internal tool_calls format to streaming delta format."""
    if not tool_calls:
        return None
    result = []
    for i, tc in enumerate(tool_calls):
        func = tc.get("function", {})
        name = func.get("name", "")
        if not name:
            # Skip malformed tool calls with missing function name
            continue
        result.append(
            ToolCallDelta(
                index=i,
                id=tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=func.get("arguments", "{}"),
                ),
            )
        )
    return result if result else None


def create_chat_chunk(
    request_id: str,
    model: str,
    created: int,
    content: str | None = None,
    thinking: str | None = None,
    role: Role | None = None,
    finish_reason: FinishReason | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> ChatCompletionChunk:
    """Create a chat completion chunk for streaming.

    Args:
        request_id: Unique request identifier.
        model: Model name for response.
        created: Unix timestamp.
        content: Main response content delta.
        thinking: Thinking content delta (llm-infer extension, scaffolded for
            future streaming separation - currently only used in non-streaming).
        role: Message role (only set on first chunk).
        finish_reason: Finish reason (only set on final chunk).
        tool_calls: Tool calls (only set on final chunk when model calls tools).
    """
    tool_call_deltas = _convert_tool_calls_to_deltas(tool_calls)
    delta = ChatCompletionChunkDelta(
        role=role, content=content, thinking=thinking, tool_calls=tool_call_deltas
    )
    return ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
    )


def create_completion_chunk(
    request_id: str,
    model: str,
    created: int,
    text: str,
    finish_reason: FinishReason | None = None,
) -> CompletionChunk:
    """Create a legacy completion chunk for streaming."""
    return CompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            CompletionChunkChoice(
                index=0,
                text=text,
                finish_reason=finish_reason,
            )
        ],
    )


async def stream_chat_completion(
    request_id: str,
    model: str,
    token_iterator: AsyncIterator[str],
    get_finish_reason: Callable[[], FinishReason | None],
) -> AsyncIterator[str]:
    """Stream chat completion as SSE events."""
    created = int(time.time())

    # First chunk: role announcement
    yield format_sse_event(
        create_chat_chunk(
            request_id=request_id, model=model, created=created, role=Role.ASSISTANT
        ).model_dump_json()
    )

    # Content chunks
    async for token in token_iterator:
        yield format_sse_event(
            create_chat_chunk(
                request_id=request_id, model=model, created=created, content=token
            ).model_dump_json()
        )

    # Final chunk with finish_reason
    yield format_sse_event(
        create_chat_chunk(
            request_id=request_id,
            model=model,
            created=created,
            finish_reason=get_finish_reason(),
        ).model_dump_json()
    )
    yield format_sse_done()


def stream_chat_completion_sync(
    request_id: str,
    model: str,
    token_iterator: Iterator[str],
    get_finish_reason: Callable[[], FinishReason | None],
) -> Iterator[str]:
    """Stream chat completion as SSE events (sync version)."""
    created = int(time.time())

    # First chunk: role announcement
    yield format_sse_event(
        create_chat_chunk(
            request_id=request_id, model=model, created=created, role=Role.ASSISTANT
        ).model_dump_json()
    )

    # Content chunks
    for token in token_iterator:
        yield format_sse_event(
            create_chat_chunk(
                request_id=request_id, model=model, created=created, content=token
            ).model_dump_json()
        )

    # Final chunk with finish_reason
    yield format_sse_event(
        create_chat_chunk(
            request_id=request_id,
            model=model,
            created=created,
            finish_reason=get_finish_reason(),
        ).model_dump_json()
    )
    yield format_sse_done()


def stream_completion_sync(
    request_id: str,
    model: str,
    token_iterator: Iterator[str],
    get_finish_reason: Callable[[], FinishReason | None],
) -> Iterator[str]:
    """
    Stream legacy completion as SSE events (sync version).

    Args:
        request_id: Unique request identifier
        model: Model name for response
        token_iterator: Iterator yielding tokens
        get_finish_reason: Callback to get finish reason when done

    Yields:
        SSE-formatted strings
    """
    created = int(time.time())

    # Content chunks
    for token in token_iterator:
        chunk = create_completion_chunk(
            request_id=request_id,
            model=model,
            created=created,
            text=token,
        )
        yield format_sse_event(chunk.model_dump_json())

    # Final chunk with finish_reason
    finish_reason = get_finish_reason()
    final_chunk = create_completion_chunk(
        request_id=request_id,
        model=model,
        created=created,
        text="",
        finish_reason=finish_reason,
    )
    yield format_sse_event(final_chunk.model_dump_json())
    yield format_sse_done()
