"""Public API schemas and client for llm-infer.

This module provides clean public exports of OpenAI-compatible schemas and
a multi-backend client, enabling downstream projects (proxies, frontends) to
import without reaching into internal module paths.

Usage:
    from llm_infer.api import ChatCompletionRequest, ChatCompletionResponse, ChatMessage

    # Client usage
    from llm_infer.api import LLMClient, ChatResponse

    with LLMClient.openai(base_url="http://localhost:8000/v1") as client:
        response = client.chat([{"role": "user", "content": "Hello"}])
"""

from llm_infer.client import Backend, ChatResponse, LLMClient
from llm_infer.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FinishReason,
    ModelInfo,
    ModelList,
    Role,
)

__all__ = [
    # Enums
    "Role",
    "FinishReason",
    # Chat messages
    "ChatMessage",
    # Chat completions (non-streaming)
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    # Chat completions (streaming)
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    # Legacy completions
    "CompletionRequest",
    "CompletionResponse",
    "CompletionChoice",
    "CompletionChunk",
    "CompletionChunkChoice",
    # Embeddings
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingObject",
    "EmbeddingUsage",
    # Models
    "ModelInfo",
    "ModelList",
    # Client
    "Backend",
    "ChatResponse",
    "LLMClient",
]
