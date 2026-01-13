"""OpenAI-compatible Pydantic models for request/response schemas."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Role(str, Enum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reason generation stopped."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


# =============================================================================
# Chat Messages
# =============================================================================


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Role
    content: str
    name: str | None = None


# =============================================================================
# Chat Completions
# =============================================================================


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body."""

    model: str = Field(
        ..., description="Model to use (accepted but ignored internally)"
    )
    messages: list[ChatMessage] = Field(..., min_length=1)

    # Generation parameters
    max_tokens: int | None = Field(None, ge=1, le=4096)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=1)  # Only n=1 supported
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)

    # Accepted but ignored (for SDK compatibility)
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def validate_unsupported(self) -> "ChatCompletionRequest":
        """Reject requests with unsupported features that would silently fail."""
        if self.n != 1:
            raise ValueError("Only n=1 is supported")
        return self


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: FinishReason | None
    logprobs: None = None


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Non-streaming chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: str | None = None


# =============================================================================
# Streaming Chat Completions
# =============================================================================


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in streaming response."""

    role: Role | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A single streaming chunk choice."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: FinishReason | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    """SSE streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    system_fingerprint: str | None = None


# =============================================================================
# Legacy Completions
# =============================================================================


class CompletionRequest(BaseModel):
    """POST /v1/completions request body (legacy endpoint)."""

    model: str = Field(
        ..., description="Model to use (accepted but ignored internally)"
    )
    prompt: str | list[str] = Field(...)

    # Generation parameters
    max_tokens: int = Field(16, ge=1, le=4096)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    echo: bool = False
    suffix: str | None = None

    # Accepted but ignored
    best_of: int | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = None
    user: str | None = None

    @model_validator(mode="after")
    def validate_unsupported(self) -> "CompletionRequest":
        """Reject requests with unsupported features."""
        if self.n != 1:
            raise ValueError("Only n=1 is supported")
        return self


class CompletionChoice(BaseModel):
    """Legacy completion choice."""

    index: int
    text: str
    finish_reason: FinishReason | None
    logprobs: None = None


class CompletionResponse(BaseModel):
    """Legacy completion response."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: ChatCompletionUsage


# =============================================================================
# Streaming Completions
# =============================================================================


class CompletionChunkChoice(BaseModel):
    """Streaming completion choice."""

    index: int
    text: str
    finish_reason: FinishReason | None = None
    logprobs: None = None


class CompletionChunk(BaseModel):
    """SSE streaming chunk for legacy completions."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]


# =============================================================================
# Models Endpoint
# =============================================================================


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "local"


class ModelList(BaseModel):
    """GET /v1/models response."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


# =============================================================================
# Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """POST /v1/embeddings request body."""

    model: str = Field(
        ..., description="Model to use (accepted but ignored internally)"
    )
    input: str | list[str] = Field(..., description="Text(s) to embed")
    encoding_format: Literal["float"] = Field(
        "float", description="Encoding format for embeddings (only float supported)"
    )
    dimensions: int | None = Field(
        None, description="Number of dimensions (for Matryoshka embeddings)"
    )
    user: str | None = None  # Accepted but ignored


class EmbeddingObject(BaseModel):
    """A single embedding result."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage for embedding request."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """POST /v1/embeddings response."""

    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingUsage
