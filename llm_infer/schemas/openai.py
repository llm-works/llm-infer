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
    DEVELOPER = "developer"  # OpenAI o1/o3/GPT-5 reasoning models


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


class TextContentPart(BaseModel):
    """Text content part in a multi-part message."""

    type: Literal["text"] = "text"
    text: str


class ImageUrlDetail(BaseModel):
    """Image URL with optional detail level."""

    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImageContentPart(BaseModel):
    """Image content part (accepted but not processed by local backends)."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlDetail


# Union of content part types - text is extracted, others are ignored
ContentPart = TextContentPart | ImageContentPart


def extract_text_from_content(
    content: str | list[ContentPart] | list[dict[str, Any]] | None,
) -> str | None:
    """Extract text from content, handling both string and array formats.

    For array format, extracts and joins all text parts. Non-text parts
    (images, audio) are silently ignored as local backends don't support them.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    # Array format: extract text parts (handles both Pydantic models and dicts)
    text_parts = []
    for part in content:
        if isinstance(part, TextContentPart):
            text_parts.append(part.text)
        elif isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(part.get("text", ""))
    return "".join(text_parts) if text_parts else None


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Role
    # String or array format. Array accepts dicts for forward compatibility with new content types.
    content: str | list[ContentPart] | list[dict[str, Any]] | None = None
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
        description="Model to use. Reserved values 'default' and 'auto' route to the "
        "base model. Any other value is treated as an adapter key (OpenAI compatibility).",
    )
    messages: list[ChatMessage] = Field(..., min_length=1)

    # Generation parameters
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(
        None,
        ge=1,
        description="Max tokens for reasoning models (alias for max_tokens)",
    )
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
    # OpenAI reasoning model parameters (accepted but ignored)
    reasoning_effort: str | None = None
    store: bool | None = None
    metadata: dict[str, str] | None = None
    service_tier: str | None = None
    stream_options: dict[str, Any] | None = None

    # llm-infer extension: LoRA adapter selection
    adapter: str | None = Field(
        None,
        description="LoRA adapter key to use for this request (llm-infer extension)",
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

    @model_validator(mode="after")
    def validate_no_mixed_roles(self) -> "ChatCompletionRequest":
        """Reject requests that mix system and developer roles."""
        has_system = any(msg.role == Role.SYSTEM for msg in self.messages)
        has_developer = any(msg.role == Role.DEVELOPER for msg in self.messages)
        if has_system and has_developer:
            raise ValueError("Cannot use both 'system' and 'developer' roles")
        return self


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: FinishReason | None
    logprobs: None = None


class CompletionTokensDetails(BaseModel):
    """Detailed completion token breakdown (reasoning models)."""

    reasoning_tokens: int = 0


class PromptTokensDetails(BaseModel):
    """Detailed prompt token breakdown."""

    cached_tokens: int = 0


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Optional detail fields for reasoning models (None for local backends)
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class AdapterInfoResponse(BaseModel):
    """LoRA adapter information (llm-infer extension).

    Tracks which adapter was requested, which was actually used,
    and metadata for verification.
    """

    requested: str | None = None  # The adapter the client requested
    actual: str | None = None  # The adapter the engine actually used
    fallback: bool = False  # True if actual != requested
    mtime: str | None = None  # ISO-8601 modification time of weights file
    md5: str | None = None  # First 12 chars of MD5 hash of weights file


class ChatCompletionResponse(BaseModel):
    """Non-streaming chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: str | None = None
    # llm-infer extensions
    adapter: AdapterInfoResponse | None = None


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
    # llm-infer extensions (only present in final chunk if adapter was requested)
    adapter: AdapterInfoResponse | None = None


# =============================================================================
# Legacy Completions
# =============================================================================


class CompletionRequest(BaseModel):
    """POST /v1/completions request body (legacy endpoint)."""

    model: str = Field(
        ...,
        description="Model to use. Reserved values 'default' and 'auto' route to the "
        "base model. Any other value is treated as an adapter key (OpenAI compatibility).",
    )
    prompt: str | list[str] = Field(...)

    # Generation parameters
    max_tokens: int = Field(16, ge=1)
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
    adapter: str | None = Field(
        None,
        description="LoRA adapter key to use for this request (llm-infer extension)",
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
    # llm-infer extensions
    adapter: AdapterInfoResponse | None = None


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
    # llm-infer extensions (only present in final chunk if adapter was requested)
    adapter: AdapterInfoResponse | None = None


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
