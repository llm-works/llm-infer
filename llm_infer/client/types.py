"""Type definitions for the LLM client.

This module defines the request/response types for the client, including
llm-infer specific extensions like thinking content and tool calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..schemas.openai import (
    ChatCompletionUsage,
    FinishReason,
    ToolCall,
)
from .backends.provider import Provider

if TYPE_CHECKING:
    from .backends import Backend

__all__ = [
    "AdapterInfo",
    "ChatRequest",
    "ChatResponse",
    "ChatStream",
    "ChatStreamAsync",
    "ChatStreamSync",
    "LLMCallbacks",
    "Provider",
    "ResponseHolder",
    "RouterChatStream",
    "RouterChatStreamSync",
]


@runtime_checkable
class ChatStreamAsync(Protocol):
    """Protocol for async chat streams with response access."""

    def __aiter__(self) -> ChatStreamAsync: ...
    async def __anext__(self) -> str: ...

    @property
    def response(self) -> ChatResponse | None:
        """The ChatResponse after stream completes. None while streaming."""
        ...


@runtime_checkable
class ChatStreamSyncProto(Protocol):
    """Protocol for sync chat streams with response access."""

    def __iter__(self) -> ChatStreamSyncProto: ...
    def __next__(self) -> str: ...

    @property
    def response(self) -> ChatResponse | None:
        """The ChatResponse after stream completes. None while streaming."""
        ...


@dataclass
class ChatRequest:
    """Request for a chat completion.

    Captures the parameters sent to a chat endpoint. Used for logging,
    debugging, and strategy inspection.

    Attributes:
        messages: List of chat messages.
        model: Model to use.
        system: System prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        tools: Tool definitions for function calling.
        tool_choice: Control tool use.
        think: Enable thinking mode.
        adapter: LoRA adapter name.
        extra: Backend-specific parameters.
        context: User-provided context passed to callbacks (cost tracking, tracing).
    """

    messages: list[dict[str, Any]]
    model: str | None = None
    system: str | None = None
    temperature: float = 1.0
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    think: bool | None = None
    adapter: str | None = None
    extra: dict[str, Any] | None = None
    context: dict[str, Any] | None = None


@dataclass
class AdapterInfo:
    """LoRA adapter information for a completion request.

    Tracks which adapter was requested, which was actually used by the
    inference engine, and metadata for verification.

    Attributes:
        requested: The adapter name the client requested.
        actual: The adapter name the engine actually used (from response).
        fallback: True if actual != requested (adapter wasn't available).
        mtime: ISO-8601 modification time of the adapter weights file.
        md5: First 12 chars of MD5 hash of the adapter weights file.
    """

    requested: str | None = None
    actual: str | None = None
    fallback: bool = False
    mtime: str | None = None
    md5: str | None = None


@dataclass
class ChatResponse:
    """Response from a chat completion request.

    This dataclass represents the response from any backend, providing a
    unified interface regardless of whether the backend is OpenAI-compatible
    or Anthropic.

    Attributes:
        content: The generated text content. May be empty if only tool_calls
            are present.
        usage: Token usage statistics (prompt, completion, total).
        finish_reason: Why generation stopped (stop, length, tool_calls, etc).
        model: The model that generated the response.
        provider: Backend/provider that generated the response (see Provider enum).
        raw: Raw response from the provider API. Contains provider-specific
            fields not normalized into the standard response (e.g., cost_in_usd_ticks).

    llm-infer Extensions:
        thinking: Extracted thinking/reasoning content from <think> blocks.
            Only present when think mode is enabled.
        tool_calls: List of tool/function calls made by the model. Present
            when the model invokes tools during generation.
        adapter: LoRA adapter info including requested/actual adapter and
            verification metadata. Only present if adapter was requested.
    """

    content: str
    usage: ChatCompletionUsage | None = None
    finish_reason: FinishReason | None = None
    model: str | None = None
    provider: str | None = None
    raw: dict[str, Any] | None = None
    # llm-infer extensions
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = field(default=None)
    adapter: AdapterInfo | None = None

    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


# Callback type aliases
LLMRequestCallback = Callable[["ChatRequest", int], None]
LLMResponseCallback = Callable[["ChatRequest", "ChatResponse"], None]
LLMErrorCallback = Callable[["ChatRequest", Exception], None]


@dataclass
class LLMCallbacks:
    """Callbacks for LLM request lifecycle events.

    Configure callbacks to observe request/response flow for cost tracking,
    logging, tracing, or metrics collection.

    Attributes:
        on_request: Called before each request attempt. Args: (request, retry).
            retry is 0 for first attempt, 1+ for retries after transient errors.
        on_response: Called after successful response. Args: (request, response).
            For streaming, fires after stream completes.
        on_error: Called after failed request. Args: (request, exception).
    """

    on_request: LLMRequestCallback | None = None
    on_response: LLMResponseCallback | None = None
    on_error: LLMErrorCallback | None = None


class ChatStream:
    """Async streaming wrapper with per-request response access.

    Wraps an async token iterator and captures the response when iteration
    completes. Use `response` property after consuming the stream to get
    usage statistics and metadata.

    This is concurrent-safe: each stream instance holds its own response,
    avoiding the race conditions of shared `last_response` state.

    Example:
        stream = await client.chat_stream_async(messages)
        async for token in stream:
            print(token, end="")
        print(f"Tokens used: {stream.response.usage.total_tokens}")
    """

    def __init__(self, inner: AsyncIterator[str], backend: Backend) -> None:
        self._inner = inner
        self._backend = backend
        self._response: ChatResponse | None = None

    def __aiter__(self) -> ChatStream:
        return self

    async def __anext__(self) -> str:
        try:
            return await self._inner.__anext__()
        except StopAsyncIteration:
            # Capture response immediately - no await between here and assignment
            self._response = self._backend.last_response
            raise

    @property
    def response(self) -> ChatResponse | None:
        """The ChatResponse after stream completes. None while streaming."""
        return self._response


class ChatStreamSync:
    """Sync streaming wrapper with per-request response access.

    Wraps a sync token iterator and captures the response when iteration
    completes. Use `response` property after consuming the stream to get
    usage statistics and metadata.

    Example:
        stream = client.chat_stream(messages)
        for token in stream:
            print(token, end="")
        print(f"Tokens used: {stream.response.usage.total_tokens}")
    """

    def __init__(self, inner: Iterator[str], backend: Backend) -> None:
        self._inner = inner
        self._backend = backend
        self._response: ChatResponse | None = None

    def __iter__(self) -> ChatStreamSync:
        return self

    def __next__(self) -> str:
        try:
            return next(self._inner)
        except StopIteration:
            self._response = self._backend.last_response
            raise

    @property
    def response(self) -> ChatResponse | None:
        """The ChatResponse after stream completes. None while streaming."""
        return self._response


class ResponseHolder:
    """Mutable container for capturing response during iteration.

    Used by router streams where the response comes from different backends
    depending on fallback behavior.
    """

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: ChatResponse | None = None


class RouterChatStream:
    """Async streaming wrapper for router with response capture.

    Like ChatStream, but uses a ResponseHolder that the router's generator
    populates directly, enabling proper response capture with fallback logic.
    """

    def __init__(
        self, inner: AsyncIterator[str], response_holder: ResponseHolder
    ) -> None:
        self._inner = inner
        self._holder = response_holder

    def __aiter__(self) -> RouterChatStream:
        return self

    async def __anext__(self) -> str:
        return await self._inner.__anext__()

    @property
    def response(self) -> ChatResponse | None:
        """The ChatResponse after stream completes. None while streaming."""
        return self._holder.value


class RouterChatStreamSync:
    """Sync streaming wrapper for router with response capture.

    Like ChatStreamSync, but uses a ResponseHolder that the router's generator
    populates directly, enabling proper response capture with fallback logic.
    """

    def __init__(self, inner: Iterator[str], response_holder: ResponseHolder) -> None:
        self._inner = inner
        self._holder = response_holder

    def __iter__(self) -> RouterChatStreamSync:
        return self

    def __next__(self) -> str:
        return next(self._inner)

    @property
    def response(self) -> ChatResponse | None:
        """The ChatResponse after stream completes. None while streaming."""
        return self._holder.value
