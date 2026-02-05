"""OpenAI-compatible Pydantic models for request/response schemas."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Role(StrEnum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    """Reason generation stopped."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


# =============================================================================
# Tool/Function Calling Types
# =============================================================================


class FunctionDefinition(BaseModel):
    """Function definition within a tool."""

    name: str = Field(..., description="The name of the function to call")
    description: str | None = Field(
        None, description="Description of what the function does"
    )
    parameters: dict[str, Any] | None = Field(
        None, description="JSON Schema for function parameters"
    )


class Tool(BaseModel):
    """Tool definition for function calling."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    """Specific function to call."""

    name: str


class ToolChoiceObject(BaseModel):
    """Tool choice specifying a specific function."""

    type: Literal["function"] = "function"
    function: ToolChoiceFunction


# tool_choice can be "auto", "none", "required", or a specific tool object
ToolChoice = Literal["auto", "none", "required"] | ToolChoiceObject


class FunctionCall(BaseModel):
    """Function call in assistant message."""

    name: str = Field(..., description="Name of the function to call")
    arguments: str = Field(..., description="JSON string of function arguments")


class ToolCall(BaseModel):
    """Tool call in assistant message."""

    id: str = Field(..., description="Unique ID for this tool call")
    type: Literal["function"] = "function"
    function: FunctionCall


# =============================================================================
# Response Format Types (Structured Output)
# =============================================================================


class JSONSchema(BaseModel):
    """JSON Schema definition for structured output."""

    name: str = Field(..., description="Name for the schema")
    description: str | None = Field(None, description="Optional description")
    schema_: dict[str, Any] = Field(
        ..., alias="schema", description="JSON Schema object"
    )
    strict: bool | None = Field(None, description="Enable strict schema adherence")


class ResponseFormatText(BaseModel):
    """Response format for unstructured text (default)."""

    type: Literal["text"] = "text"


class ResponseFormatJSONObject(BaseModel):
    """Response format for valid JSON output (no schema enforcement)."""

    type: Literal["json_object"] = "json_object"


class ResponseFormatJSONSchema(BaseModel):
    """Response format for JSON with strict schema enforcement."""

    type: Literal["json_schema"] = "json_schema"
    json_schema: JSONSchema


ResponseFormat = (
    ResponseFormatText | ResponseFormatJSONObject | ResponseFormatJSONSchema
)


# =============================================================================
# Chat Messages
# =============================================================================


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Role
    content: str | None = None  # Can be None when tool_calls present
    name: str | None = None
    # Tool calling fields
    tool_calls: list[ToolCall] | None = Field(
        None, description="Tool calls made by the assistant (assistant messages only)"
    )
    tool_call_id: str | None = Field(
        None,
        description="ID of the tool call this message responds to (tool messages only)",
    )
    # llm-infer extension: separated thinking content
    thinking: str | None = Field(
        None,
        description="Thinking/reasoning content extracted from <think> blocks "
        "(llm-infer extension, only present in assistant messages when think mode is active)",
    )


# =============================================================================
# Chat Completions
# =============================================================================


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body."""

    model: str = Field(
        ...,
        description="Model to use. Use 'default' or 'auto' to use the loaded model. "
        "Server ignores this value and uses its loaded model.",
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

    # Tool/function calling
    tools: list[Tool] | None = Field(
        None, description="List of tools the model may call"
    )
    tool_choice: ToolChoice | None = Field(
        None,
        description="Controls tool use: 'auto' (default), 'none', 'required', "
        "or specific tool",
    )

    # Structured output
    response_format: ResponseFormat | None = Field(
        None,
        description="Controls output format: 'text' (default), 'json_object' "
        "(valid JSON), or 'json_schema' (strict schema enforcement)",
    )

    # Accepted but ignored (for SDK compatibility)
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None
    seed: int | None = None

    # llm-infer extension: LoRA adapter selection
    adapter_id: str | None = Field(
        None,
        description="Name of LoRA adapter to use for this request (llm-infer extension)",
    )

    # llm-infer extension: Think mode control
    think: bool | None = Field(
        None,
        description="Enable/disable think mode. When None, uses model config default. "
        "Server injects model-appropriate suffix and normalizes output tags to "
        "canonical form from model config (llm-infer extension)",
    )

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


class ToolCallDelta(BaseModel):
    """Tool call delta for streaming responses.

    In streaming, tool calls are sent incrementally with an index to identify
    which tool call is being updated.
    """

    index: int = Field(..., description="Index of the tool call being streamed")
    id: str | None = Field(None, description="Tool call ID (sent in first chunk)")
    type: Literal["function"] | None = None
    function: FunctionCall | None = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in streaming response."""

    role: Role | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    # llm-infer extension: separated thinking content
    thinking: str | None = None


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
        ...,
        description="Model to use. Use 'default' or 'auto' to use the loaded model. "
        "Server ignores this value and uses its loaded model.",
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

    # llm-infer extension: LoRA adapter selection
    adapter_id: str | None = Field(
        None,
        description="Name of LoRA adapter to use for this request (llm-infer extension)",
    )

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
        ...,
        description="Model to use. Use 'default' or 'auto' to use the loaded model. "
        "Server ignores this value and uses its loaded model.",
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
