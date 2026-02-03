"""Anthropic Claude backend implementation.

This backend uses the official Anthropic Python SDK to interact with Claude
models. It requires the anthropic package to be installed:

    pip install llm-infer[anthropic]

Key differences from OpenAI-compatible backends:
- System messages are passed as a separate parameter, not in messages
- Thinking mode is not yet supported (extended_thinking requires different API structure)
- Tool calling uses Anthropic's native format
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from llm_infer.client.backends.base import Backend
from llm_infer.client.exceptions import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_infer.client.types import ChatResponse
from llm_infer.schemas.openai import (
    ChatCompletionUsage,
    FinishReason,
    FunctionCall,
    ToolCall,
)

if TYPE_CHECKING:
    import anthropic


class AnthropicBackend(Backend):
    """Backend for Anthropic Claude API.

    This backend uses the official Anthropic SDK. It translates between
    the common Backend interface and Anthropic's native API format.

    Requires: pip install llm-infer[anthropic]

    Example:
        # Sync usage
        with AnthropicBackend(api_key="sk-ant-...") as backend:
            response = backend.chat([{"role": "user", "content": "Hello"}])
            print(response.content)

        # Async streaming
        async with AnthropicBackend() as backend:
            async for token in backend.chat_stream_async(messages):
                print(token, end="")

    Note:
        - System messages should be passed via the `system` parameter,
          not in the messages list.
        - adapter_id is not supported (Anthropic-specific feature).
        - think mode is currently not supported (raises NotImplementedError).
    """

    def __init__(
        self,
        lg: Logger,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the backend."""
        try:
            import anthropic as anthropic_module
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install llm-infer[anthropic]"
            ) from e

        self._lg = lg
        self._anthropic = anthropic_module
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._last_response: ChatResponse | None = None

        # Sync client created eagerly
        self._client: anthropic.Anthropic = anthropic_module.Anthropic(
            api_key=api_key, timeout=timeout
        )
        # Async client created lazily
        self._async_client: anthropic.AsyncAnthropic | None = None
        self._api_key = api_key

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
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat completion request (sync)."""
        _ = adapter_id  # Not supported for Anthropic
        request_kwargs = self._prepare_request(
            messages,
            model,
            temperature,
            max_tokens,
            system,
            think,
            tools,
            tool_choice,
            **kwargs,
        )
        with self._handle_errors():
            response = self._client.messages.create(**request_kwargs)
        result = self._parse_response(response, model or self._model)
        self._last_response = result
        return result

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Send a streaming chat completion request (sync)."""
        _ = adapter_id
        request_kwargs = self._prepare_request(
            messages,
            model,
            temperature,
            max_tokens,
            system,
            think,
            tools,
            tool_choice,
            stream=True,
            **kwargs,
        )
        state = _StreamState()
        with self._handle_errors():
            with self._client.messages.stream(**request_kwargs) as stream:
                for event in stream:
                    token = self._process_stream_event(event, state)
                    if token:
                        yield token
                self._finalize_stream_state(state, stream.get_final_message())
        self._last_response = state.to_response(model or self._model)

    # =========================================================================
    # Async methods
    # =========================================================================

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat completion request (async)."""
        _ = adapter_id
        request_kwargs = self._prepare_request(
            messages,
            model,
            temperature,
            max_tokens,
            system,
            think,
            tools,
            tool_choice,
            **kwargs,
        )
        client = self._get_async_client()
        async with self._handle_errors_async():
            response = await client.messages.create(**request_kwargs)
        result = self._parse_response(response, model or self._model)
        self._last_response = result
        return result

    async def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request (async)."""
        _ = adapter_id
        request_kwargs = self._prepare_request(
            messages,
            model,
            temperature,
            max_tokens,
            system,
            think,
            tools,
            tool_choice,
            stream=True,
            **kwargs,
        )
        state = _StreamState()
        client = self._get_async_client()
        async with self._handle_errors_async():
            async with client.messages.stream(**request_kwargs) as stream:
                async for event in stream:
                    token = self._process_stream_event(event, state)
                    if token:
                        yield token
                self._finalize_stream_state(state, await stream.get_final_message())
        self._last_response = state.to_response(model or self._model)

    # =========================================================================
    # Resource management
    # =========================================================================

    def _get_async_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the async client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = self._anthropic.AsyncAnthropic(
                api_key=self._api_key, timeout=self._timeout
            )
        return self._async_client

    def close(self) -> None:
        """Close sync client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close all clients (sync and async)."""
        self._client.close()
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    @contextmanager
    def _handle_errors(self) -> Generator[None, None, None]:
        """Context manager to translate Anthropic exceptions to backend errors."""
        try:
            yield
        except self._anthropic.APIConnectionError as e:
            raise BackendUnavailableError("Failed to connect to Anthropic API") from e
        except self._anthropic.APITimeoutError as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except self._anthropic.APIStatusError as e:
            raise BackendRequestError(
                f"Anthropic API error: {e.message}", status_code=e.status_code
            ) from e

    @asynccontextmanager
    async def _handle_errors_async(self) -> AsyncGenerator[None, None]:
        """Async context manager to translate Anthropic exceptions to backend errors."""
        try:
            yield
        except self._anthropic.APIConnectionError as e:
            raise BackendUnavailableError("Failed to connect to Anthropic API") from e
        except self._anthropic.APITimeoutError as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except self._anthropic.APIStatusError as e:
            raise BackendRequestError(
                f"Anthropic API error: {e.message}", status_code=e.status_code
            ) from e

    # =========================================================================
    # Factory
    # =========================================================================

    @classmethod
    def from_config(cls, lg: Logger, config: dict[str, Any]) -> AnthropicBackend:
        """Create backend from configuration dict."""
        return cls(
            lg=lg,
            model=config.get("model", "claude-sonnet-4-20250514"),
            api_key=config.get("api_key"),
            max_tokens=config.get("max_tokens", 4096),
            timeout=config.get("timeout", 120.0),
        )

    # =========================================================================
    # Request preparation
    # =========================================================================

    def _prepare_request(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        system: str | None,
        think: bool | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare request kwargs for Anthropic API."""
        converted_messages = self._convert_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": converted_messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature,
        }

        if system:
            request_kwargs["system"] = system
        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)
        if tool_choice:
            self._apply_tool_choice(request_kwargs, tool_choice)

        if think:
            raise NotImplementedError(
                "think mode is not yet supported for Anthropic backend; "
                "extended_thinking requires different API structure"
            )

        for key, value in kwargs.items():
            if value is not None and key not in ("stream",):
                request_kwargs[key] = value

        return request_kwargs

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                continue  # System messages passed via system param
            converted = self._convert_single_message(msg)
            if converted:
                result.append(converted)
        return result

    def _convert_single_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a single message to Anthropic format."""
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }
                ],
            }
        elif role == "assistant" and msg.get("tool_calls"):
            return self._convert_assistant_with_tools(msg, content)
        else:
            return {"role": role, "content": content}

    def _convert_assistant_with_tools(
        self, msg: dict[str, Any], content: str
    ) -> dict[str, Any]:
        """Convert assistant message with tool calls."""
        content_blocks: list[dict[str, Any]] = []
        if content:
            content_blocks.append({"type": "text", "text": content})
        for tc in msg["tool_calls"]:
            # Parse arguments from JSON string to dict for Anthropic
            args_raw = tc.get("function", {}).get("arguments", "{}")
            args_parsed: dict[str, Any]
            if args_raw is None:
                args_parsed = {}
            elif isinstance(args_raw, str):
                try:
                    args_parsed = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args_parsed = {}
            else:
                args_parsed = args_raw
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "input": args_parsed,
                }
            )
        return {"role": "assistant", "content": content_blocks}

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tools to Anthropic format."""
        result = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                result.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
        return result

    def _apply_tool_choice(
        self, request_kwargs: dict[str, Any], tool_choice: str | dict[str, Any]
    ) -> None:
        """Apply tool_choice to request kwargs."""
        if tool_choice == "auto":
            request_kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "none":
            request_kwargs.pop("tools", None)
        elif tool_choice == "required":
            request_kwargs["tool_choice"] = {"type": "any"}
        elif isinstance(tool_choice, dict) and "function" in tool_choice:
            request_kwargs["tool_choice"] = {
                "type": "tool",
                "name": tool_choice["function"].get("name", ""),
            }

    # =========================================================================
    # Response parsing
    # =========================================================================

    def _parse_response(self, response: Any, model: str) -> ChatResponse:
        """Parse Anthropic response to ChatResponse."""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            self._parse_content_block(block, content_parts, thinking_parts, tool_calls)

        return ChatResponse(
            content="".join(content_parts),
            usage=self._create_usage(getattr(response, "usage", None)),
            finish_reason=self._map_stop_reason(response.stop_reason),
            model=response.model or model,
            thinking="".join(thinking_parts) if thinking_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _parse_content_block(
        self,
        block: Any,
        content_parts: list[str],
        thinking_parts: list[str],
        tool_calls: list[ToolCall],
    ) -> None:
        """Parse a content block from response."""
        if block.type == "text":
            content_parts.append(block.text)
        elif block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "tool_use":
            tool_calls.append(self._create_tool_call(block))

    def _create_tool_call(self, block: Any) -> ToolCall:
        """Create a ToolCall from an Anthropic tool_use block."""
        return ToolCall(
            id=block.id,
            type="function",
            function=FunctionCall(
                name=block.name,
                arguments=block.input
                if isinstance(block.input, str)
                else json.dumps(block.input),
            ),
        )

    def _create_usage(self, usage: Any) -> ChatCompletionUsage | None:
        """Create ChatCompletionUsage from Anthropic usage."""
        if usage is None:
            return None
        return ChatCompletionUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )

    # =========================================================================
    # Stream processing
    # =========================================================================

    def _process_stream_event(self, event: Any, state: _StreamState) -> str | None:
        """Process a single stream event, returning text delta if present."""
        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            return self._process_delta(event, state)
        elif event_type == "content_block_stop":
            self._process_block_stop(event, state)
        elif event_type == "message_delta":
            # Capture usage from message_delta events (authoritative source)
            usage = getattr(event, "usage", None)
            if usage:
                state.usage = self._create_usage(usage)
        return None

    def _process_delta(self, event: Any, state: _StreamState) -> str | None:
        """Process a content_block_delta event."""
        delta = getattr(event, "delta", None)
        if not delta:
            return None

        delta_type = getattr(delta, "type", None)
        if delta_type == "text_delta":
            text = getattr(delta, "text", "")
            state.content_parts.append(text)
            return text
        elif delta_type == "thinking_delta":
            state.thinking_parts.append(getattr(delta, "thinking", ""))
        return None

    def _process_block_stop(self, event: Any, state: _StreamState) -> None:
        """Process a content_block_stop event."""
        block = getattr(event, "content_block", None)
        if block and getattr(block, "type", None) == "tool_use":
            state.tool_calls.append(self._create_tool_call(block))

    def _finalize_stream_state(self, state: _StreamState, final_message: Any) -> None:
        """Finalize stream state with usage and finish reason."""
        if final_message:
            # Prefer usage from message_delta events (already captured in state.usage)
            # Fall back to final_message.usage only if not captured during streaming
            if state.usage is None:
                state.usage = self._create_usage(getattr(final_message, "usage", None))
            state.finish_reason = self._map_stop_reason(final_message.stop_reason)

    def _map_stop_reason(self, stop_reason: str | None) -> FinishReason | None:
        """Map Anthropic stop_reason to FinishReason."""
        if stop_reason is None:
            return None
        mapping = {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALLS,
        }
        return mapping.get(stop_reason)


class _StreamState:
    """Accumulates state while processing stream events."""

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.tool_calls: list[ToolCall] = []
        self.usage: ChatCompletionUsage | None = None
        self.finish_reason: FinishReason | None = None

    def to_response(self, model: str) -> ChatResponse:
        """Convert accumulated state to ChatResponse."""
        return ChatResponse(
            content="".join(self.content_parts),
            usage=self.usage,
            finish_reason=self.finish_reason,
            model=model,
            thinking="".join(self.thinking_parts) if self.thinking_parts else None,
            tool_calls=self.tool_calls if self.tool_calls else None,
        )
