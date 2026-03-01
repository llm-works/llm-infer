"""FastAPI router for OpenAI-compatible endpoints."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ....response.parsers.think import ThinkTagNormalizer, extract_thinking
from ....schemas.openai import (
    AdapterInfoResponse,
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
    FinishReason,
    FunctionCall,
    ModelInfo,
    ModelList,
    Role,
    ToolCall,
)
from ..errors import raise_for_error_status
from .mappers import (
    chat_request_to_internal,
    completion_request_to_internal,
    determine_finish_reason,
    generate_tool_call_id,
    normalize_arguments,
    resolve_think_mode,
)
from .streaming_generators import ChatStreamingGenerator, CompletionStreamingGenerator

if TYPE_CHECKING:
    from ....models.config import ModelConfig


def _build_completion_usage(response: Any) -> ChatCompletionUsage:
    """Build usage stats from response."""
    return ChatCompletionUsage(
        prompt_tokens=response.prompt_tokens or 0,
        completion_tokens=response.completion_tokens or 0,
        total_tokens=(response.prompt_tokens or 0) + (response.completion_tokens or 0),
    )


def _create_normalizer(
    think: bool | None, model_config: ModelConfig | None
) -> ThinkTagNormalizer | None:
    """Create tag normalizer if think mode is active."""
    if not model_config:
        return None
    effective_think = resolve_think_mode(think, model_config)
    if not effective_think:
        return None
    think_config = model_config.think
    return ThinkTagNormalizer(think_config.tags_open, think_config.tags_close)


def _normalize_response(
    text: str, think: bool | None, model_config: ModelConfig | None
) -> str:
    """Normalize think tags in response text if think mode is active."""
    if not text:
        return text
    normalizer = _create_normalizer(think, model_config)
    if not normalizer:
        return text
    return normalizer.process(text) + normalizer.flush()


def _get_think_tags(
    model_config: ModelConfig | None,
) -> tuple[list[str], list[str]]:
    """Get think tags from model config or return defaults."""
    if model_config and model_config.think:
        return model_config.think.tags_open, model_config.think.tags_close
    return ["<think>", "<thinking>"], ["</think>", "</thinking>"]


def _extract_and_separate_thinking(
    text: str, think: bool | None, model_config: ModelConfig | None
) -> tuple[str | None, str]:
    """Extract thinking content and separate from main content.

    Returns (thinking, content) where thinking is None if think mode is inactive
    or no think blocks were found.
    """
    if not text:
        return None, text

    # Check if think mode is active
    effective_think = resolve_think_mode(think, model_config)
    if not effective_think:
        return None, text

    # First normalize tags, then extract
    normalized = _normalize_response(text, think, model_config)
    open_tags, close_tags = _get_think_tags(model_config)
    return extract_thinking(normalized, open_tags, close_tags)


def _convert_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
) -> list[ToolCall] | None:
    """Convert internal tool_calls format to OpenAI schema objects."""
    if not tool_calls:
        return None
    result = []
    for tc in tool_calls:
        # Handle both Ollama format and already-converted format
        func = tc.get("function", {})
        name = func.get("name", "")
        if not name:
            # Skip malformed tool calls with missing function name
            continue
        result.append(
            ToolCall(
                id=tc.get("id", generate_tool_call_id()),
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=normalize_arguments(func.get("arguments")),
                ),
            )
        )
    return result if result else None


def _build_chat_response(
    request_id: str,
    model_name: str,
    content: str | None,
    thinking: str | None,
    finish_reason: FinishReason,
    response: Any,
    tool_calls: list[ToolCall] | None = None,
) -> ChatCompletionResponse:
    """Build chat completion response object."""
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role=Role.ASSISTANT,
                    content=content,
                    thinking=thinking,
                    tool_calls=tool_calls,
                    tool_call_id=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=_build_completion_usage(response),
        adapter=_build_adapter_info(response),
    )


def _determine_chat_finish_reason(
    response: Any, body: ChatCompletionRequest, has_tool_calls: bool
) -> FinishReason:
    """Determine finish reason for chat completion."""
    max_tokens_reached = (
        response.completion_tokens is not None
        and body.max_tokens is not None
        and response.completion_tokens >= body.max_tokens
    )
    return determine_finish_reason(
        is_eos=not max_tokens_reached,
        max_tokens_reached=max_tokens_reached,
        has_tool_calls=has_tool_calls,
    )


def _build_adapter_info(response: Any) -> AdapterInfoResponse | None:
    """Build AdapterInfoResponse from internal response."""
    adapter = getattr(response, "adapter", None)
    if adapter is None:
        return None
    return AdapterInfoResponse(
        requested=adapter.requested,
        actual=adapter.actual,
        fallback=adapter.fallback,
        mtime=adapter.mtime,
        md5=adapter.md5,
    )


def _build_adapter_fallback_headers(response: Any) -> dict[str, str]:
    """Build adapter fallback headers if fallback occurred."""
    adapter = getattr(response, "adapter", None)
    if adapter is not None and adapter.fallback:
        return {
            "X-Adapter-Fallback": "true",
            "X-Adapter-Requested": adapter.requested or "",
        }
    return {}


async def _handle_chat_non_streaming(
    request_id: str,
    body: ChatCompletionRequest,
    model_name: str,
    ipc: Any,
    model_config: ModelConfig | None = None,
) -> ChatCompletionResponse | JSONResponse:
    """Handle non-streaming chat completion request."""
    internal_request = chat_request_to_internal(body, request_id, model_config)
    response = await ipc.submit(request_id, internal_request)
    raise_for_error_status(response)

    thinking, content = _extract_and_separate_thinking(
        response.result or "", body.think, model_config
    )
    tool_calls = _convert_tool_calls(getattr(response, "tool_calls", None))
    has_tool_calls = bool(tool_calls)
    finish_reason = _determine_chat_finish_reason(response, body, has_tool_calls)

    chat_response = _build_chat_response(
        request_id,
        model_name,
        content if content else None,
        thinking,
        finish_reason,
        response,
        tool_calls,
    )

    # Add adapter fallback headers if needed
    headers = _build_adapter_fallback_headers(response)
    if headers:
        return JSONResponse(
            content=chat_response.model_dump(mode="json"),
            headers=headers,
        )
    return chat_response


def _build_completion_response_obj(
    request_id: str,
    body: CompletionRequest,
    model_name: str,
    response: Any,
) -> CompletionResponse:
    """Build CompletionResponse from internal response."""
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
        adapter=_build_adapter_info(response),
    )


async def _handle_completion_non_streaming(
    request_id: str,
    body: CompletionRequest,
    model_name: str,
    ipc: Any,
) -> CompletionResponse | JSONResponse:
    """Handle non-streaming legacy completion request."""
    internal_request = completion_request_to_internal(body, request_id)
    response = await ipc.submit(request_id, internal_request)
    raise_for_error_status(response)

    completion_response = _build_completion_response_obj(
        request_id, body, model_name, response
    )

    headers = _build_adapter_fallback_headers(response)
    if headers:
        return JSONResponse(
            content=completion_response.model_dump(mode="json"),
            headers=headers,
        )
    return completion_response


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


def _handle_chat_streaming(
    request_id: str,
    body: ChatCompletionRequest,
    model_name: str,
    ipc: Any,
    model_config: ModelConfig | None,
) -> StreamingResponse:
    """Handle streaming chat completion request."""
    internal_request = chat_request_to_internal(body, request_id, model_config)
    normalizer = _create_normalizer(body.think, model_config)
    generator = ChatStreamingGenerator(request_id, model_name, ipc, normalizer)
    return StreamingResponse(
        generator.stream(internal_request),
        media_type="text/event-stream",
    )


def _register_chat_completion_routes(
    router: APIRouter, model_name: str, model_config: ModelConfig | None = None
) -> None:
    """Register chat completion endpoint."""

    @router.post("/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest, request: Request
    ) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
        ipc = request.app.state.ipc_channel
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        if body.stream:
            return _handle_chat_streaming(
                request_id, body, model_name, ipc, model_config
            )
        return await _handle_chat_non_streaming(
            request_id, body, model_name, ipc, model_config
        )


def _register_legacy_completion_routes(router: APIRouter, model_name: str) -> None:
    """Register legacy completion endpoint."""

    @router.post("/completions", response_model=None)
    async def completions(
        body: CompletionRequest, request: Request
    ) -> CompletionResponse | StreamingResponse | JSONResponse:
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


def create_openai_router(
    model_name: str, model_config: ModelConfig | None = None
) -> APIRouter:
    """Create OpenAI-compatible API router.

    Args:
        model_name: Name of the loaded model.
        model_config: Optional model config for server-side handling of system
            prompts and think mode.
    """
    router = APIRouter(tags=["OpenAI"])
    _register_model_routes(router, model_name)
    _register_chat_completion_routes(router, model_name, model_config)
    _register_legacy_completion_routes(router, model_name)
    _register_embedding_routes(router, model_name)
    return router
