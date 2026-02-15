"""SSE streaming utilities for OpenAI-compatible responses."""

import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from ....schemas.openai import (
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
from .mappers import generate_tool_call_id, normalize_arguments


def format_sse_event(data: str) -> str:
    """Format data as SSE event."""
    return f"data: {data}\n\n"


def format_sse_done() -> str:
    """Format SSE done marker."""
    return "data: [DONE]\n\n"


def _convert_tool_calls_to_deltas(
    tool_calls: list[dict[str, Any]] | None,
) -> list[ToolCallDelta] | None:
    """Convert internal tool_calls format to streaming delta format.

    Uses sequential indices (0, 1, 2...) for valid tool calls, skipping malformed
    entries. This matches OpenAI's expected format where indices are contiguous.
    """
    if not tool_calls:
        return None
    result: list[ToolCallDelta] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        if not name:
            # Skip malformed tool calls with missing function name
            continue
        result.append(
            ToolCallDelta(
                index=len(result),  # Sequential index for valid tool calls
                id=tc.get("id", generate_tool_call_id()),
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=normalize_arguments(func.get("arguments")),
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
    adapter_fallback: bool | None = None,
    adapter_requested: str | None = None,
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
        adapter_fallback: True if adapter was requested but not found (final chunk only).
        adapter_requested: The adapter that was requested (final chunk only).
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
        adapter_fallback=adapter_fallback,
        adapter_requested=adapter_requested,
    )


def create_completion_chunk(
    request_id: str,
    model: str,
    created: int,
    text: str,
    finish_reason: FinishReason | None = None,
    adapter_fallback: bool | None = None,
    adapter_requested: str | None = None,
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
        adapter_fallback=adapter_fallback,
        adapter_requested=adapter_requested,
    )


async def stream_chat_completion(
    request_id: str,
    model: str,
    token_iterator: AsyncIterator[str],
    get_finish_reason: Callable[[], FinishReason | None],
    adapter_fallback: bool | None = None,
    adapter_requested: str | None = None,
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

    # Final chunk with finish_reason and adapter fallback info
    yield format_sse_event(
        create_chat_chunk(
            request_id=request_id,
            model=model,
            created=created,
            finish_reason=get_finish_reason(),
            adapter_fallback=adapter_fallback,
            adapter_requested=adapter_requested,
        ).model_dump_json(exclude_none=True)
    )
    yield format_sse_done()


def stream_chat_completion_sync(
    request_id: str,
    model: str,
    token_iterator: Iterator[str],
    get_finish_reason: Callable[[], FinishReason | None],
    adapter_fallback: bool | None = None,
    adapter_requested: str | None = None,
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

    # Final chunk with finish_reason and adapter fallback info
    yield format_sse_event(
        create_chat_chunk(
            request_id=request_id,
            model=model,
            created=created,
            finish_reason=get_finish_reason(),
            adapter_fallback=adapter_fallback,
            adapter_requested=adapter_requested,
        ).model_dump_json(exclude_none=True)
    )
    yield format_sse_done()


def stream_completion_sync(
    request_id: str,
    model: str,
    token_iterator: Iterator[str],
    get_finish_reason: Callable[[], FinishReason | None],
    adapter_fallback: bool | None = None,
    adapter_requested: str | None = None,
) -> Iterator[str]:
    """
    Stream legacy completion as SSE events (sync version).

    Args:
        request_id: Unique request identifier
        model: Model name for response
        token_iterator: Iterator yielding tokens
        get_finish_reason: Callback to get finish reason when done
        adapter_fallback: True if adapter was requested but not found
        adapter_requested: The adapter that was requested (if fallback occurred)

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

    # Final chunk with finish_reason and adapter fallback info
    finish_reason = get_finish_reason()
    final_chunk = create_completion_chunk(
        request_id=request_id,
        model=model,
        created=created,
        text="",
        finish_reason=finish_reason,
        adapter_fallback=adapter_fallback,
        adapter_requested=adapter_requested,
    )
    yield format_sse_event(final_chunk.model_dump_json(exclude_none=True))
    yield format_sse_done()
