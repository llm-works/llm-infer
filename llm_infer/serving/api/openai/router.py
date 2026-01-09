"""FastAPI router for OpenAI-compatible endpoints."""

import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .mappers import (
    chat_request_to_internal,
    completion_request_to_internal,
    determine_finish_reason,
)
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    ModelInfo,
    ModelList,
    Role,
)
from .streaming import (
    create_chat_chunk,
    create_completion_chunk,
    format_sse_done,
    format_sse_event,
)

if TYPE_CHECKING:
    from ...dispatch.types import Request as InternalRequest


def _raise_for_error_status(response) -> None:
    """Raise HTTPException if response indicates an error."""
    from ...dispatch.types import RequestStatus

    if response.status == RequestStatus.REJECTED:
        raise HTTPException(
            status_code=503, detail=response.error or "Server at capacity"
        )
    if response.status == RequestStatus.FAILED:
        raise HTTPException(status_code=500, detail=response.error or "Internal error")


def _build_completion_usage(response) -> ChatCompletionUsage:
    """Build usage stats from response."""
    return ChatCompletionUsage(
        prompt_tokens=response.prompt_tokens or 0,
        completion_tokens=response.completion_tokens or 0,
        total_tokens=(response.prompt_tokens or 0) + (response.completion_tokens or 0),
    )


async def _handle_chat_non_streaming(
    request_id: str,
    body: ChatCompletionRequest,
    model_name: str,
    ipc,
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion request."""
    internal_request = chat_request_to_internal(body, request_id)
    response = await ipc.submit(request_id, internal_request)
    _raise_for_error_status(response)

    max_tokens_reached = (
        response.completion_tokens is not None
        and body.max_tokens is not None
        and response.completion_tokens >= body.max_tokens
    )
    finish_reason = determine_finish_reason(
        is_eos=not max_tokens_reached, max_tokens_reached=max_tokens_reached
    )

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role=Role.ASSISTANT, content=response.result or ""),
                finish_reason=finish_reason,
            )
        ],
        usage=_build_completion_usage(response),
    )


def _map_finish_reason(reason: str | None) -> FinishReason:
    """Map internal finish reason to OpenAI finish reason."""
    if reason == "length":
        return FinishReason.LENGTH
    if reason == "error":
        return FinishReason.CONTENT_FILTER
    return FinishReason.STOP


async def _stream_chat_completion(
    request_id: str,
    internal_request: "InternalRequest",
    model: str,
    ipc,
) -> AsyncIterator[str]:
    """Generate SSE stream for chat completion."""
    created = int(time.time())

    # First chunk: role announcement
    first_chunk = create_chat_chunk(
        request_id=request_id, model=model, created=created, role=Role.ASSISTANT
    )
    yield format_sse_event(first_chunk.model_dump_json())

    # Stream tokens
    finish_reason = FinishReason.STOP
    async for chunk in ipc.submit_streaming(request_id, internal_request):
        if chunk.is_final:
            finish_reason = _map_finish_reason(chunk.finish_reason)
            break
        if chunk.token:
            content_chunk = create_chat_chunk(
                request_id=request_id, model=model, created=created, content=chunk.token
            )
            yield format_sse_event(content_chunk.model_dump_json())

    # Final chunk with finish_reason
    final_chunk = create_chat_chunk(
        request_id=request_id, model=model, created=created, finish_reason=finish_reason
    )
    yield format_sse_event(final_chunk.model_dump_json())
    yield format_sse_done()


async def _handle_completion_non_streaming(
    request_id: str,
    body: CompletionRequest,
    model_name: str,
    ipc,
) -> CompletionResponse:
    """Handle non-streaming legacy completion request."""
    internal_request = completion_request_to_internal(body, request_id)
    response = await ipc.submit(request_id, internal_request)
    _raise_for_error_status(response)

    max_tokens_reached = (
        response.completion_tokens is not None
        and response.completion_tokens >= body.max_tokens
    )
    finish_reason = determine_finish_reason(
        is_eos=not max_tokens_reached, max_tokens_reached=max_tokens_reached
    )

    result_text = response.result or ""
    if body.echo:
        prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]
        result_text = prompt + result_text

    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            CompletionChoice(index=0, text=result_text, finish_reason=finish_reason)
        ],
        usage=_build_completion_usage(response),
    )


async def _stream_completion(
    request_id: str,
    internal_request: "InternalRequest",
    model: str,
    ipc,
) -> AsyncIterator[str]:
    """Generate SSE stream for legacy completion."""
    created = int(time.time())

    finish_reason = FinishReason.STOP
    async for chunk in ipc.submit_streaming(request_id, internal_request):
        if chunk.is_final:
            finish_reason = _map_finish_reason(chunk.finish_reason)
            break
        if chunk.token:
            content_chunk = create_completion_chunk(
                request_id=request_id, model=model, created=created, text=chunk.token
            )
            yield format_sse_event(content_chunk.model_dump_json())

    final_chunk = create_completion_chunk(
        request_id=request_id,
        model=model,
        created=created,
        text="",
        finish_reason=finish_reason,
    )
    yield format_sse_event(final_chunk.model_dump_json())
    yield format_sse_done()


def _create_model_info(model_name: str) -> ModelInfo:
    """Create ModelInfo for the current model."""
    return ModelInfo(id=model_name, created=int(time.time()), owned_by="local")


def _register_model_routes(router: APIRouter, model_name: str) -> None:
    """Register /models endpoints."""

    @router.get("/models", response_model=ModelList)
    async def list_models() -> ModelList:
        return ModelList(data=[_create_model_info(model_name)])

    @router.get("/models/{model_id}", response_model=ModelInfo)
    async def get_model(model_id: str) -> ModelInfo:
        if model_id != model_name:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found. Available: {model_name}",
            )
        return _create_model_info(model_name)


def _register_completion_routes(router: APIRouter, model_name: str) -> None:
    """Register completion endpoints."""

    @router.post("/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        ipc = request.app.state.ipc_channel
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        if body.stream:
            internal_request = chat_request_to_internal(body, request_id)
            return StreamingResponse(
                _stream_chat_completion(request_id, internal_request, model_name, ipc),
                media_type="text/event-stream",
            )
        return await _handle_chat_non_streaming(request_id, body, model_name, ipc)

    @router.post("/completions")
    async def completions(body: CompletionRequest, request: Request):
        ipc = request.app.state.ipc_channel
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        if body.stream:
            internal_request = completion_request_to_internal(body, request_id)
            return StreamingResponse(
                _stream_completion(request_id, internal_request, model_name, ipc),
                media_type="text/event-stream",
            )
        return await _handle_completion_non_streaming(request_id, body, model_name, ipc)


def create_openai_router(model_name: str) -> APIRouter:
    """Create OpenAI-compatible API router."""
    router = APIRouter(tags=["OpenAI"])
    _register_model_routes(router, model_name)
    _register_completion_routes(router, model_name)
    return router
