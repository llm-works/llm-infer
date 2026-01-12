"""OpenAI-compatible API layer."""

from llm_infer.schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    ModelInfo,
    ModelList,
    Role,
)

from .errors import ErrorResponse, OpenAIHTTPException, create_error_response
from .mappers import (
    chat_request_to_internal,
    completion_request_to_internal,
    determine_finish_reason,
)
from .router import create_openai_router

__all__ = [
    # Chat types
    "Role",
    "ChatMessage",
    "FinishReason",
    # Chat completions
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    # Legacy completions
    "CompletionRequest",
    "CompletionResponse",
    # Models
    "ModelInfo",
    "ModelList",
    # Router
    "create_openai_router",
    # Mappers
    "chat_request_to_internal",
    "completion_request_to_internal",
    "determine_finish_reason",
    # Errors
    "ErrorResponse",
    "OpenAIHTTPException",
    "create_error_response",
]
