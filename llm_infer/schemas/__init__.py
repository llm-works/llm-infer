"""Schema definitions for llm-infer.

This package contains pure data models (Pydantic schemas) that can be imported
without triggering circular dependencies from the serving infrastructure.
"""

from .openai import (
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
]
