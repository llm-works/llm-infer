"""OpenAI-compatible async client for consuming SSE streams.

This module provides a reusable client for calling OpenAI-compatible APIs,
with full support for SSE streaming. It enables downstream packages (proxies,
frontends) to consume streaming responses without reimplementing SSE parsing.

Usage:
    from llm_infer.client import OpenAIClient

    client = OpenAIClient(base_url="http://localhost:8000/v1")

    # Non-streaming
    response = await client.chat(messages, system="You are helpful.")
    print(response.content)

    # Streaming
    async for token in client.chat_stream(messages):
        print(token, end="", flush=True)
    print(client.last_response.usage)  # Usage stats after streaming
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    # Import for type checking only (avoids circular import at runtime)
    from llm_infer.serving.api.openai.schemas import (
        ChatCompletionUsage,
        ChatMessage,
        FinishReason,
        Role,
    )
else:
    # Runtime import from clean API path
    from llm_infer.api import (
        ChatCompletionUsage,
        ChatMessage,
        FinishReason,
        Role,
    )


@dataclass
class ChatResponse:
    """Response from a chat completion request.

    Attributes:
        content: The generated text content.
        usage: Token usage statistics (prompt, completion, total).
        finish_reason: Why generation stopped (stop, length, content_filter).
    """

    content: str
    usage: ChatCompletionUsage | None = None
    finish_reason: FinishReason | None = None


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for chat completion clients.

    This protocol defines the interface for chat clients, enabling type-safe
    dependency injection and clean mocking in tests.

    Example (mocking in tests):
        class MockChatClient:
            def __init__(self, responses: list[str]):
                self._responses = iter(responses)
                self._last_response: ChatResponse | None = None

            @property
            def last_response(self) -> ChatResponse | None:
                return self._last_response

            async def chat(self, messages, **kwargs) -> ChatResponse:
                content = next(self._responses)
                self._last_response = ChatResponse(content=content)
                return self._last_response

            async def chat_stream(self, messages, **kwargs) -> AsyncIterator[str]:
                content = next(self._responses)
                for char in content:
                    yield char
                self._last_response = ChatResponse(content=content)

        # Use in tests
        client: ChatClient = MockChatClient(["Hello!", "Goodbye!"])
    """

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats (populated after streaming)."""
        ...

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        system: str | None = None,
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat completion request."""
        ...

    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        system: str | None = None,
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request, yielding tokens."""
        ...


class OpenAIClient:
    """Async client for OpenAI-compatible APIs with SSE streaming support.

    This client handles both streaming and non-streaming chat completions,
    parsing SSE events and extracting tokens for easy consumption.

    Example:
        client = OpenAIClient(
            base_url="http://localhost:8000/v1",
            api_key="sk-...",  # Optional
            timeout=120.0,
        )

        # Non-streaming request
        response = await client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

        # Streaming request
        async for token in client.chat_stream(messages):
            print(token, end="")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Base URL for the API (e.g., "http://localhost:8000/v1").
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._last_response: ChatResponse | None = None

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats.

        This is populated after streaming completes, providing access to
        usage statistics that are only available at the end of the stream.
        """
        return self._last_response

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        system: str | None = None,
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat completion request.

        Args:
            messages: List of chat messages (ChatMessage objects or dicts).
            system: Optional system prompt (prepended to messages).
            model: Model name to use.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters passed to the API.

        Returns:
            ChatResponse with content, usage stats, and finish reason.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.RequestError: If the request fails.
        """
        url = f"{self._base_url}/chat/completions"
        built_messages = self._build_messages(messages, system)
        payload = self._build_payload(
            built_messages, model, temperature, max_tokens, stream=False, **kwargs
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=self._build_headers())
            resp.raise_for_status()
            data = resp.json()

        # Extract response components
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = _parse_finish_reason(choice.get("finish_reason"))
        usage = _parse_usage(data.get("usage"))

        response = ChatResponse(
            content=content,
            usage=usage,
            finish_reason=finish_reason,
        )
        self._last_response = response
        return response

    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        system: str | None = None,
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request.

        Yields tokens as they arrive from the server. After iteration completes,
        access `last_response` for usage statistics.

        Args:
            messages: List of chat messages (ChatMessage objects or dicts).
            system: Optional system prompt (prepended to messages).
            model: Model name to use.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters passed to the API.

        Yields:
            String tokens as they arrive.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.RequestError: If the request fails.
        """
        url = f"{self._base_url}/chat/completions"
        built_messages = self._build_messages(messages, system)
        payload = self._build_payload(
            built_messages, model, temperature, max_tokens, stream=True, **kwargs
        )

        state = _StreamState()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=self._build_headers()
            ) as resp:
                resp.raise_for_status()
                async for chunk in _parse_sse_stream(resp):
                    token = state.process_chunk(chunk)
                    if token:
                        yield token

        self._last_response = state.to_response()

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_messages(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        system: str | None,
    ) -> list[dict[str, Any]]:
        """Build messages list, prepending system prompt if provided."""
        result: list[dict[str, Any]] = []

        if system:
            result.append({"role": Role.SYSTEM.value, "content": system})

        for msg in messages:
            if isinstance(msg, ChatMessage):
                result.append({"role": msg.role.value, "content": msg.content})
            else:
                result.append(msg)

        return result

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request payload."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Pass through any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        return payload


@dataclass
class _StreamState:
    """Accumulates state while processing SSE stream chunks."""

    content: list[str] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    usage: ChatCompletionUsage | None = None

    def process_chunk(self, chunk: dict[str, Any]) -> str | None:
        """Process a single SSE chunk, returning token content if present."""
        choices = chunk.get("choices", [])
        token = None

        if choices:
            delta = choices[0].get("delta", {})
            token = delta.get("content")
            if token:
                self.content.append(token)

            chunk_finish = choices[0].get("finish_reason")
            if chunk_finish:
                self.finish_reason = _parse_finish_reason(chunk_finish)

        if "usage" in chunk:
            self.usage = _parse_usage(chunk["usage"])

        return token

    def to_response(self) -> ChatResponse:
        """Convert accumulated state to ChatResponse."""
        return ChatResponse(
            content="".join(self.content),
            usage=self.usage,
            finish_reason=self.finish_reason,
        )


async def _parse_sse_stream(
    response: httpx.Response,
) -> AsyncIterator[dict[str, Any]]:
    """Parse SSE stream from httpx response.

    SSE format:
        data: {"json": "payload"}\n\n
        data: [DONE]\n\n

    Args:
        response: The httpx streaming response.

    Yields:
        Parsed JSON objects from SSE data lines.
    """
    async for line in response.aiter_lines():
        # Skip empty lines and non-data lines
        if not line or not line.startswith("data: "):
            continue

        # Extract data after "data: " prefix
        data = line[6:]

        # Check for stream termination
        if data == "[DONE]":
            break

        # Parse and yield JSON
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            # Skip malformed chunks (matches CLI behavior)
            continue


def _parse_finish_reason(value: str | None) -> FinishReason | None:
    """Parse finish reason string to enum."""
    if value is None:
        return None
    try:
        return FinishReason(value)
    except ValueError:
        return None


def _parse_usage(data: dict[str, Any] | None) -> ChatCompletionUsage | None:
    """Parse usage dict to ChatCompletionUsage."""
    if data is None:
        return None
    return ChatCompletionUsage(
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        total_tokens=data.get("total_tokens", 0),
    )


__all__ = ["ChatClient", "ChatResponse", "OpenAIClient"]
