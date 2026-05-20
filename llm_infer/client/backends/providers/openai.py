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

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from appinfra.log import Logger

from ....schemas.openai import (
    ChatCompletionUsage,
    CompletionTokensDetails,
    FinishReason,
    FunctionCall,
    PromptTokensDetails,
    Role,
    ToolCall,
)
from ...errors import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from ...types import AdapterInfo, ChatRequest, ChatResponse, ResponseHolder
from ..base import Backend
from ..context import BackendContext
from ..embedding import Backend as EmbeddingBackend
from ..embedding import BatchEmbeddingResult, EmbeddingResult
from ..mixins import AsyncRequestTrackingMixin
from ..provider import ProviderDetector


class OpenAICompatibleBackend(AsyncRequestTrackingMixin, Backend):
    """Backend for OpenAI-compatible APIs."""

    def __init__(
        self,
        lg: Logger,
        name: str,
        ctx: BackendContext | None = None,
        default_model: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            lg: Logger instance.
            name: Backend name (for discovery/routing).
            ctx: Backend context with rate limiter, backoff, and timeouts.
            default_model: Default model if not specified per-request.
            base_url: Base URL for the API.
            api_key: API key for authentication.
        """
        super().__init__(lg, name, ctx, default_model)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._provider = ProviderDetector.detect(base_url, api_key)
        self._last_response: ChatResponse | None = None
        self._client = httpx.Client(timeout=self._ctx.request_timeout)
        self._async_client: httpx.AsyncClient | None = None
        self._init_async_tracking()

    @property
    def provider(self) -> str:
        """Detected provider for this backend."""
        return self._provider.value

    @property
    def last_response(self) -> ChatResponse | None:
        """Last response with usage stats."""
        return self._last_response

    # =========================================================================
    # Sync methods
    # =========================================================================

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming chat completion request (sync)."""
        url, payload = self._prepare_request(request, stream=False)
        data = self._execute_sync(url, payload)
        response = self._parse_chat_response(data, request.model or self.default_model)
        self._last_response = response
        return response

    def chat_stream(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> Iterator[str]:
        """Send a streaming chat completion request (sync)."""
        url, payload = self._prepare_request(request, stream=True)
        state = _StreamState()
        for chunk in self._execute_stream_sync(url, payload):
            token = state.process_chunk(chunk)
            if token:
                yield token
        response = state.to_response(request.model or self.default_model, self.provider)
        self._last_response = response
        if holder is not None:
            holder.value = response

    # =========================================================================
    # Async methods
    # =========================================================================

    async def chat_async(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming chat completion request (async)."""
        url, payload = self._prepare_request(request, stream=False)
        data = await self._execute_async(url, payload)
        response = self._parse_chat_response(data, request.model or self.default_model)
        self._last_response = response
        return response

    async def chat_stream_async(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request (async)."""
        url, payload = self._prepare_request(request, stream=True)
        state = _StreamState()
        async for chunk in self._execute_stream_async(url, payload):
            token = state.process_chunk(chunk)
            if token:
                yield token
        response = state.to_response(request.model or self.default_model, self.provider)
        self._last_response = response
        if holder is not None:
            holder.value = response

    # =========================================================================
    # Model discovery
    # =========================================================================

    def list_models(self) -> list[str]:
        """List available models from this backend via /v1/models endpoint."""
        if self._ctx.rate_limiter is not None:
            self._ctx.rate_limiter.next()
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
                f"Request timed out after {self._ctx.request_timeout}s"
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
        if self._ctx.rate_limiter is not None:
            self._ctx.rate_limiter.next()
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
                f"Request timed out after {self._ctx.request_timeout}s"
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
        if self._ctx.rate_limiter is not None:
            self._ctx.rate_limiter.next()
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
                f"Request timed out after {self._ctx.request_timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            e.response.read()  # Must read body for streaming responses
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    async def _execute_async(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute async request with error translation."""
        if self._ctx.rate_limiter is not None:
            await asyncio.to_thread(self._ctx.rate_limiter.next)
        self._acquire_async_request()
        try:
            client = self._get_async_client()
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
                f"Request timed out after {self._ctx.request_timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e
        finally:
            self._release_async_request()

    async def _execute_stream_async(
        self, url: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute async streaming request with error translation."""
        if self._ctx.rate_limiter is not None:
            await asyncio.to_thread(self._ctx.rate_limiter.next)
        self._acquire_async_request()
        try:
            client = self._get_async_client()
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
                f"Request timed out after {self._ctx.request_timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            await e.response.aread()  # Must read body for streaming responses
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e
        finally:
            self._release_async_request()

    # =========================================================================
    # Resource management
    # =========================================================================

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._ctx.request_timeout)
        return self._async_client

    def close(self) -> None:
        """Close sync HTTP client.

        Note: If async operations were used, call aclose() instead to also
        close the async client.
        """
        self._client.close()

    async def aclose(self) -> None:
        """Close all HTTP clients (sync and async).

        Waits for any in-flight async requests to complete before closing.
        New requests will fail immediately once close is requested.
        """
        await self._drain_async_requests()
        self._client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    # =========================================================================
    # Request preparation
    # =========================================================================

    def _prepare_request(
        self, request: ChatRequest, stream: bool
    ) -> tuple[str, dict[str, Any]]:
        """Prepare URL and payload for request."""
        url = f"{self._base_url}/chat/completions"
        built_messages = self._build_messages(request.messages, request.system)
        payload = self._build_payload(request, built_messages, stream)
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
        self, request: ChatRequest, messages: list[dict[str, Any]], stream: bool
    ) -> dict[str, Any]:
        """Build the request payload."""
        payload: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": stream,
        }
        if stream:
            # Request usage stats in final chunk. May break non-compliant backends
            # that reject unknown params, but required for billing/cost tracking.
            payload["stream_options"] = {"include_usage": True}
        self._add_optional_params(payload, request)
        # Add extra params, filtering out reserved keys to prevent override
        if request.extra:
            # Extract extra_body contents as top-level keys (OpenAI SDK convention)
            extra_body = request.extra.get("extra_body")
            if extra_body:
                for key, value in extra_body.items():
                    if value is not None and key not in self._RESERVED_KEYS:
                        payload[key] = value
            for key, value in request.extra.items():
                if key == "extra_body":
                    continue
                if value is not None and key not in self._RESERVED_KEYS:
                    payload[key] = value
        return payload

    def _add_optional_params(
        self, payload: dict[str, Any], request: ChatRequest
    ) -> None:
        """Add optional parameters to payload."""
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.think is not None:
            payload["think"] = request.think
        if request.adapter is not None:
            payload["adapter"] = request.adapter

    # =========================================================================
    # Response parsing
    # =========================================================================

    def _parse_chat_response(
        self, data: dict[str, Any], model: str | None
    ) -> ChatResponse:
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
            provider=self.provider,
            raw=data,
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
    raw: dict[str, Any] | None = None

    def process_chunk(self, chunk: dict[str, Any]) -> str | None:
        """Process a single SSE chunk, returning token content if present."""
        choices = chunk.get("choices", [])
        token = None

        if choices:
            token = self._process_choice(choices[0])

        if "usage" in chunk:
            self.usage = _parse_usage(chunk["usage"])
            self.raw = chunk

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

    def to_response(
        self, model: str | None, provider: str | None = None
    ) -> ChatResponse:
        """Convert accumulated state to ChatResponse."""
        tool_calls = self._finalize_tool_calls()
        return ChatResponse(
            content="".join(self.content),
            usage=self.usage,
            finish_reason=self.finish_reason,
            model=model,
            provider=provider,
            raw=self.raw,
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

    prompt_details = None
    if prompt_data := data.get("prompt_tokens_details"):
        prompt_details = PromptTokensDetails(
            cached_tokens=prompt_data.get("cached_tokens") or 0,
        )

    completion_details = None
    if completion_data := data.get("completion_tokens_details"):
        completion_details = CompletionTokensDetails(
            reasoning_tokens=completion_data.get("reasoning_tokens") or 0,
        )

    return ChatCompletionUsage(
        prompt_tokens=data.get("prompt_tokens") or 0,
        completion_tokens=data.get("completion_tokens") or 0,
        total_tokens=data.get("total_tokens") or 0,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
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


# =============================================================================
# Embedding Backend
# =============================================================================


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI-compatible embedding backend.

    Works with OpenAI, Azure OpenAI, and any API following the /v1/embeddings format.

    Example:
        backend = OpenAIEmbeddingBackend(
            lg=logger,
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="text-embedding-3-small",
        )
        result = backend.embed("Hello world")
    """

    def __init__(
        self,
        lg: Logger,
        base_url: str,
        model: str,
        api_key: str | None = None,
        ctx: BackendContext | None = None,
    ) -> None:
        """Initialize OpenAI-compatible embedding backend.

        Args:
            lg: Logger instance.
            base_url: Base URL (e.g., "https://api.openai.com/v1").
            model: Model name (e.g., "text-embedding-3-small").
            api_key: Optional API key for Authorization header.
            ctx: Backend context with rate limiter and timeouts.
        """
        super().__init__(lg, model, ctx)
        self._base_url = base_url.rstrip("/")

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(timeout=self._ctx.request_timeout, headers=headers)
        self._async_client: httpx.AsyncClient | None = None
        self._headers = headers

    @property
    def provider(self) -> str:
        return "openai"

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self._ctx.request_timeout, headers=self._headers
            )
        return self._async_client

    # =========================================================================
    # HTTP execution
    # =========================================================================

    def _execute_sync(
        self,
        texts: str | list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Execute sync request with error translation."""
        url = f"{self._base_url}/embeddings"
        payload: dict[str, Any] = {"model": model or self._model, "input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._ctx.request_timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    async def _execute_async(
        self,
        texts: str | list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Execute async request with error translation."""
        url = f"{self._base_url}/embeddings"
        payload: dict[str, Any] = {"model": model or self._model, "input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        client = self._get_async_client()
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._ctx.request_timeout}s"
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
    # Response parsing
    # =========================================================================

    def _parse_single(
        self,
        data: dict[str, Any],
        model: str | None = None,
        requested_dims: int | None = None,
    ) -> EmbeddingResult:
        """Parse response for single embedding."""
        try:
            embedding = data["data"][0]["embedding"]
            actual_dims = len(embedding)
        except (KeyError, IndexError, TypeError) as e:
            raise BackendRequestError(f"Malformed response: {e}") from e
        if requested_dims is not None and actual_dims != requested_dims:
            raise BackendRequestError(
                f"Requested {requested_dims} dimensions but got {actual_dims}"
            )
        return EmbeddingResult(
            embedding=embedding,
            model=data.get("model", model or self._model),
            dimensions=actual_dims,
            prompt_tokens=data.get("usage", {}).get("prompt_tokens"),
        )

    def _extract_batch_embeddings(
        self, items: list[dict[str, Any]], requested_dims: int | None
    ) -> tuple[list[list[float]], int]:
        """Extract embeddings from sorted batch items, validate dimensions."""
        embeddings = []
        actual_dims = 0
        for i, item in enumerate(sorted(items, key=lambda x: x["index"])):
            embedding = item["embedding"]
            dims = len(embedding)
            if requested_dims is not None and dims != requested_dims:
                raise BackendRequestError(
                    f"Requested {requested_dims} dimensions but got {dims}"
                )
            if i == 0:
                actual_dims = dims
            embeddings.append(embedding)
        return embeddings, actual_dims

    def _parse_batch(
        self,
        data: dict[str, Any],
        num_texts: int,
        model: str | None = None,
        requested_dims: int | None = None,
    ) -> BatchEmbeddingResult:
        """Parse batch embedding response."""
        try:
            embeddings_data = data["data"]
        except (KeyError, TypeError) as e:
            raise BackendRequestError(f"Malformed response: {e}") from e
        if len(embeddings_data) != num_texts:
            raise BackendRequestError(
                f"Expected {num_texts} embeddings, got {len(embeddings_data)}"
            )

        try:
            embeddings, actual_dims = self._extract_batch_embeddings(
                embeddings_data, requested_dims
            )
        except (KeyError, TypeError) as e:
            raise BackendRequestError(f"Malformed response: {e}") from e

        return BatchEmbeddingResult(
            embeddings=embeddings,
            model=data.get("model", model or self._model),
            dimensions=actual_dims,
            size=num_texts,
            total_prompt_tokens=data.get("usage", {}).get("prompt_tokens"),
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        self._wait_rate_limit()
        data = self._execute_sync(text, model, dimensions)
        return self._parse_single(data, model, dimensions)

    def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> BatchEmbeddingResult:
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=model or self._model,
                dimensions=0,
                size=0,
                total_prompt_tokens=0,
            )
        self._wait_rate_limit()
        data = self._execute_sync(texts, model, dimensions)
        return self._parse_batch(data, len(texts), model, dimensions)

    async def embed_async(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        await self._wait_rate_limit_async()
        data = await self._execute_async(text, model, dimensions)
        return self._parse_single(data, model, dimensions)

    async def embed_batch_async(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> BatchEmbeddingResult:
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=model or self._model,
                dimensions=0,
                size=0,
                total_prompt_tokens=0,
            )
        await self._wait_rate_limit_async()
        data = await self._execute_async(texts, model, dimensions)
        return self._parse_batch(data, len(texts), model, dimensions)

    # =========================================================================
    # Token counting
    # =========================================================================

    def _get_tokenizer(self) -> Any:
        """Get tiktoken encoding for this model."""
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for token counting. "
                "Install with: pip install tiktoken"
            ) from e

        try:
            return tiktoken.encoding_for_model(self._model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        enc = self._get_tokenizer()
        return len(enc.encode(text))

    def count_tokens_batch(self, texts: list[str]) -> int:
        if not texts:
            return 0
        enc = self._get_tokenizer()
        return sum(len(enc.encode(text)) for text in texts)

    async def count_tokens_async(self, text: str) -> int:
        return self.count_tokens(text)

    async def count_tokens_batch_async(self, texts: list[str]) -> int:
        return self.count_tokens_batch(texts)

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        self._client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
