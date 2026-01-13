"""FastAPI router for OpenAI-compatible endpoints."""

import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_infer.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ModelInfo,
    ModelList,
    Role,
)

from ..errors import raise_for_error_status
from .mappers import (
    chat_request_to_internal,
    completion_request_to_internal,
    determine_finish_reason,
)
from .streaming_generators import ChatStreamingGenerator, CompletionStreamingGenerator

if TYPE_CHECKING:
    pass


def _build_completion_usage(response: Any) -> ChatCompletionUsage:
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
    ipc: Any,
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion request."""
    internal_request = chat_request_to_internal(body, request_id)
    response = await ipc.submit(request_id, internal_request)
    raise_for_error_status(response)

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


async def _handle_completion_non_streaming(
    request_id: str,
    body: CompletionRequest,
    model_name: str,
    ipc: Any,
) -> CompletionResponse:
    """Handle non-streaming legacy completion request."""
    internal_request = completion_request_to_internal(body, request_id)
    response = await ipc.submit(request_id, internal_request)
    raise_for_error_status(response)

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

    @router.post("/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest, request: Request
    ) -> ChatCompletionResponse | StreamingResponse:
        ipc = request.app.state.ipc_channel
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        if body.stream:
            internal_request = chat_request_to_internal(body, request_id)
            generator = ChatStreamingGenerator(request_id, model_name, ipc)
            return StreamingResponse(
                generator.stream(internal_request),
                media_type="text/event-stream",
            )
        return await _handle_chat_non_streaming(request_id, body, model_name, ipc)

    @router.post("/completions", response_model=None)
    async def completions(
        body: CompletionRequest, request: Request
    ) -> CompletionResponse | StreamingResponse:
        ipc = request.app.state.ipc_channel
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        if body.stream:
            internal_request = completion_request_to_internal(body, request_id)
            generator = CompletionStreamingGenerator(request_id, model_name, ipc)
            return StreamingResponse(
                generator.stream(internal_request),
                media_type="text/event-stream",
            )
        return await _handle_completion_non_streaming(request_id, body, model_name, ipc)


def _build_embedding_response(response: Any, model_name: str) -> EmbeddingResponse:
    """Build OpenAI-compatible embedding response from internal response."""
    embedding_objects = [
        EmbeddingObject(embedding=emb, index=i)
        for i, emb in enumerate(response.embeddings or [])
    ]
    return EmbeddingResponse(
        data=embedding_objects,
        model=model_name,
        usage=EmbeddingUsage(
            prompt_tokens=response.total_tokens,
            total_tokens=response.total_tokens,
        ),
    )


async def _handle_embedding_request(
    body: EmbeddingRequest, ipc: Any, model_name: str
) -> EmbeddingResponse:
    """Process embedding request and return response."""
    from ...dispatch.types import EmbeddingRequest as InternalEmbeddingRequest

    request_id = f"emb-{uuid.uuid4().hex[:24]}"
    inputs = [body.input] if isinstance(body.input, str) else list(body.input)
    internal_request = InternalEmbeddingRequest(
        id=request_id, inputs=inputs, dimensions=body.dimensions
    )
    response = await ipc.submit(request_id, internal_request)
    raise_for_error_status(response)
    return _build_embedding_response(response, model_name)


def _register_embedding_routes(router: APIRouter, model_name: str) -> None:
    """Register embedding endpoints."""

    @router.post("/embeddings", response_model=EmbeddingResponse)
    async def embeddings(body: EmbeddingRequest, request: Request) -> EmbeddingResponse:
        """Generate embeddings for input text(s)."""
        ipc = request.app.state.ipc_channel
        return await _handle_embedding_request(body, ipc, model_name)


def create_openai_router(model_name: str) -> APIRouter:
    """Create OpenAI-compatible API router."""
    router = APIRouter(tags=["OpenAI"])
    _register_model_routes(router, model_name)
    _register_completion_routes(router, model_name)
    _register_embedding_routes(router, model_name)
    return router
