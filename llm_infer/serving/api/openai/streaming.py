"""SSE streaming utilities for OpenAI-compatible responses."""

import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING

from llm_infer.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    CompletionChunk,
    CompletionChunkChoice,
    FinishReason,
    Role,
)

if TYPE_CHECKING:
    pass


def format_sse_event(data: str) -> str:
    """Format data as SSE event."""
    return f"data: {data}\n\n"


def format_sse_done() -> str:
    """Format SSE done marker."""
    return "data: [DONE]\n\n"


def create_chat_chunk(
    request_id: str,
    model: str,
    created: int,
    content: str | None = None,
    role: Role | None = None,
    finish_reason: FinishReason | None = None,
) -> ChatCompletionChunk:
    """Create a chat completion chunk for streaming."""
    delta = ChatCompletionChunkDelta(role=role, content=content)
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
