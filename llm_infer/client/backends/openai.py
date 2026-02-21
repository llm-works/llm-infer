"""OpenAI-compatible backend implementation.

This backend works with any OpenAI-compatible API, including:
- OpenAI API
- llm-infer server
- vLLM
- Ollama (with OpenAI compatibility layer)
- Any other OpenAI-compatible server

llm-infer Extensions:
    - adapter: LoRA adapter selection for vLLM
    - think: Thinking mode with <think> block extraction
    - tools/tool_choice: Function calling support
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from appinfra.log import Logger

from ...schemas.openai import (
    ChatCompletionUsage,
    FinishReason,
    FunctionCall,
    Role,
    ToolCall,
)
from ..exceptions import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from ..types import AdapterInfo, ChatResponse
from .base import Backend


class OpenAICompatibleBackend(Backend):
    """Backend for OpenAI-compatible APIs."""

    def __init__(
        self,
        lg: Logger,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the backend."""
        self._lg = lg
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._last_response: ChatResponse | None = None
        self._client = httpx.Client(timeout=timeout)
        self._async_client: httpx.AsyncClient | None = None

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats."""
        return self._last_response

    # =========================================================================
    # Sync methods
    # =========================================================================

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat completion request (sync)."""
        url, payload = self._prepare_request(
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            stream=False,
            **kwargs,
        )
        data = self._execute_sync(url, payload)
        response = self._parse_chat_response(data, model or self._model)
        self._last_response = response
        return response

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Send a streaming chat completion request (sync)."""
        url, payload = self._prepare_request(
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            stream=True,
            **kwargs,
        )
        state = _StreamState()
        for chunk in self._execute_stream_sync(url, payload):
            token = state.process_chunk(chunk)
            if token:
                yield token
        self._last_response = state.to_response(model or self._model)

    # =========================================================================
    # Async methods
    # =========================================================================

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat completion request (async)."""
        url, payload = self._prepare_request(
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            stream=False,
            **kwargs,
        )
        data = await self._execute_async(url, payload)
        response = self._parse_chat_response(data, model or self._model)
        self._last_response = response
        return response

    async def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request (async)."""
        url, payload = self._prepare_request(
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            stream=True,
            **kwargs,
        )
        state = _StreamState()
        async for chunk in self._execute_stream_async(url, payload):
            token = state.process_chunk(chunk)
            if token:
                yield token
        self._last_response = state.to_response(model or self._model)

    # =========================================================================
    # Model discovery
    # =========================================================================

    def list_models(self) -> list[str]:
        """List available models from this backend via /v1/models endpoint."""
        url = f"{self._base_url}/models"
        try:
            resp = self._client.get(url, headers=self._build_headers())
            resp.raise_for_status()
            data = resp.json()
            models: list[str] = [m["id"] for m in data.get("data", [])]
            return models
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except (json.JSONDecodeError, KeyError) as e:
            raise BackendRequestError(f"Invalid models response: {e}") from e

    # =========================================================================
    # Request execution
    # =========================================================================

    def _execute_sync(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute sync request with error translation."""
        try:
            resp = self._client.post(url, json=payload, headers=self._build_headers())
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    def _execute_stream_sync(
        self, url: str, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Execute sync streaming request with error translation."""
        try:
            with self._client.stream(
                "POST", url, json=payload, headers=self._build_headers()
            ) as resp:
                resp.raise_for_status()
                yield from self._parse_sse_stream_sync(resp)
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    async def _execute_async(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute async request with error translation."""
        client = self._get_async_client()
        try:
            resp = await client.post(url, json=payload, headers=self._build_headers())
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    async def _execute_stream_async(
        self, url: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute async streaming request with error translation."""
        client = self._get_async_client()
        try:
            async with client.stream(
                "POST", url, json=payload, headers=self._build_headers()
            ) as resp:
                resp.raise_for_status()
                async for chunk in self._parse_sse_stream_async(resp):
                    yield chunk
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    # =========================================================================
    # Resource management
    # =========================================================================

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout)
        return self._async_client

    def close(self) -> None:
        """Close sync HTTP client.

        Note: If async operations were used, call aclose() instead to also
        close the async client.
        """
        self._client.close()

    async def aclose(self) -> None:
        """Close all HTTP clients (sync and async)."""
        self._client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    def from_config(cls, lg: Logger, config: dict[str, Any]) -> OpenAICompatibleBackend:
        """Create backend from configuration dict."""
        return cls(
            lg=lg,
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            model=config.get("model", "default"),
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 120.0),
        )

    # =========================================================================
    # Request preparation
    # =========================================================================

    def _prepare_request(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        think: bool | None,
        adapter: str | None,
        stream: bool,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Prepare URL and payload for request."""
        url = f"{self._base_url}/chat/completions"
        built_messages = self._build_messages(messages, system)
        payload = self._build_payload(
            built_messages,
            model or self._model,
            temperature,
            max_tokens,
            stream,
            tools,
            tool_choice,
            think,
            adapter,
            **kwargs,
        )
        return url, payload

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_messages(
        self, messages: list[dict[str, Any]], system: str | None
    ) -> list[dict[str, Any]]:
        """Build messages list, prepending system prompt if provided."""
        if not system:
            return list(messages)
        return [{"role": Role.SYSTEM.value, "content": system}, *messages]

    # Reserved keys that cannot be overridden via kwargs
    _RESERVED_KEYS = frozenset(
        {
            "model",
            "messages",
            "temperature",
            "stream",
            "max_tokens",
            "adapter",
            "think",
            "tools",
            "tool_choice",
        }
    )

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        think: bool | None,
        adapter: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request payload."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        self._add_optional_params(
            payload, max_tokens, tools, tool_choice, think, adapter
        )
        # Add extra kwargs, filtering out reserved keys to prevent override
        for key, value in kwargs.items():
            if value is not None and key not in self._RESERVED_KEYS:
                payload[key] = value
        return payload

    def _add_optional_params(
        self,
        payload: dict[str, Any],
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        think: bool | None,
        adapter: str | None,
    ) -> None:
        """Add optional parameters to payload."""
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if think is not None:
            payload["think"] = think
        if adapter is not None:
            payload["adapter_id"] = adapter

    # =========================================================================
    # Response parsing
    # =========================================================================

    def _parse_chat_response(self, data: dict[str, Any], model: str) -> ChatResponse:
        """Parse API response data into ChatResponse."""
        choices = data.get("choices", [])
        if not choices:
            raise BackendRequestError("API returned empty choices array")

        choice = choices[0]
        message = choice.get("message", {})
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))

        return ChatResponse(
            content=message.get("content") or "",
            usage=_parse_usage(data.get("usage")),
            finish_reason=_parse_finish_reason(choice.get("finish_reason")),
            model=data.get("model", model),
            thinking=message.get("thinking"),
            tool_calls=tool_calls,
            adapter=_parse_adapter_info(data.get("adapter")),
        )

    def _parse_tool_calls(
        self, raw_tool_calls: list[dict[str, Any]] | None
    ) -> list[ToolCall] | None:
        """Parse tool calls from response."""
        if not raw_tool_calls:
            return None
        return [
            ToolCall(
                id=tc["id"],
                type="function",
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"].get("arguments") or "",
                ),
            )
            for tc in raw_tool_calls
        ]

    # =========================================================================
    # SSE parsing
    # =========================================================================

    def _parse_sse_stream_sync(
        self, response: httpx.Response
    ) -> Iterator[dict[str, Any]]:
        """Parse SSE stream from sync httpx response."""
        for line in response.iter_lines():
            chunk = self._parse_sse_line(line)
            if chunk is None:
                break
            if isinstance(chunk, dict):
                yield chunk

    async def _parse_sse_stream_async(
        self, response: httpx.Response
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse SSE stream from async httpx response."""
        async for line in response.aiter_lines():
            chunk = self._parse_sse_line(line)
            if chunk is None:
                break
            if isinstance(chunk, dict):
                yield chunk

    def _parse_sse_line(self, line: str) -> dict[str, Any] | bool | None:
        """Parse a single SSE line. Returns None for [DONE], False to skip, dict for data."""
        if not line or not line.startswith("data:"):
            return False
        # Per SSE spec, space after colon is optional and stripped if present
        data = line[5:]
        if data.startswith(" "):
            data = data[1:]
        if data == "[DONE]":
            return None
        try:
            result: dict[str, Any] = json.loads(data)
            return result
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid SSE JSON: {e}") from e


@dataclass
class _StreamState:
    """Accumulates state while processing SSE stream chunks."""

    content: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    _tool_call_buffer: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: FinishReason | None = None
    usage: ChatCompletionUsage | None = None
    adapter: AdapterInfo | None = None

    def process_chunk(self, chunk: dict[str, Any]) -> str | None:
        """Process a single SSE chunk, returning token content if present."""
        choices = chunk.get("choices", [])
        token = None

        if choices:
            token = self._process_choice(choices[0])

        if "usage" in chunk:
            self.usage = _parse_usage(chunk["usage"])

        # Adapter info (present in final chunk if adapter was requested)
        if "adapter" in chunk:
            self.adapter = _parse_adapter_info(chunk["adapter"])

        return token

    def _process_choice(self, choice: dict[str, Any]) -> str | None:
        """Process a single choice from chunk."""
        delta = choice.get("delta", {})
        token = None

        content = delta.get("content")
        if content:
            self.content.append(content)
            token = content

        thinking = delta.get("thinking")
        if thinking:
            self.thinking.append(thinking)

        tool_call_deltas = delta.get("tool_calls")
        if tool_call_deltas:
            self._process_tool_call_deltas(tool_call_deltas)

        chunk_finish = choice.get("finish_reason")
        if chunk_finish:
            self.finish_reason = _parse_finish_reason(chunk_finish)

        return token

    def _process_tool_call_deltas(self, deltas: list[dict[str, Any]]) -> None:
        """Process incremental tool call updates."""
        for delta in deltas:
            idx = delta.get("index", 0)
            if idx not in self._tool_call_buffer:
                self._tool_call_buffer[idx] = {
                    "id": delta.get("id", ""),
                    "function": {"name": "", "arguments": ""},
                }
            self._update_tool_call_buffer(self._tool_call_buffer[idx], delta)

    def _update_tool_call_buffer(
        self, buf: dict[str, Any], delta: dict[str, Any]
    ) -> None:
        """Update a tool call buffer with delta."""
        if delta.get("id"):
            buf["id"] = delta["id"]
        func = delta.get("function", {})
        if func.get("name"):
            buf["function"]["name"] = func["name"]
        if func.get("arguments"):
            buf["function"]["arguments"] += func["arguments"]

    def to_response(self, model: str) -> ChatResponse:
        """Convert accumulated state to ChatResponse."""
        tool_calls = self._finalize_tool_calls()
        return ChatResponse(
            content="".join(self.content),
            usage=self.usage,
            finish_reason=self.finish_reason,
            model=model,
            thinking="".join(self.thinking) if self.thinking else None,
            tool_calls=tool_calls,
            adapter=self.adapter,
        )

    def _finalize_tool_calls(self) -> list[ToolCall] | None:
        """Finalize tool calls from buffer."""
        if not self._tool_call_buffer:
            return None
        return [
            ToolCall(
                id=buf["id"],
                type="function",
                function=FunctionCall(
                    name=buf["function"]["name"],
                    arguments=buf["function"]["arguments"],
                ),
            )
            for buf in (
                self._tool_call_buffer[i] for i in sorted(self._tool_call_buffer)
            )
        ]


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
        prompt_tokens=data.get("prompt_tokens") or 0,
        completion_tokens=data.get("completion_tokens") or 0,
        total_tokens=data.get("total_tokens") or 0,
    )


def _parse_adapter_info(data: dict[str, Any] | None) -> AdapterInfo | None:
    """Parse adapter dict to AdapterInfo."""
    if data is None:
        return None
    return AdapterInfo(
        requested=data.get("requested"),
        actual=data.get("actual"),
        fallback=data.get("fallback", False),
        mtime=data.get("mtime"),
        md5=data.get("md5"),
    )
