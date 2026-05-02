"""SAIA backend adapter for LLMClient.

This module provides an adapter that allows LLMClient (and subclasses) to be
used as a SAIABackend, enabling integration with the llm-saia verb vocabulary.

Usage:
    from llm_infer.client import Factory, SAIAAdapter
    from llm_saia import SAIA

    factory = Factory(lg)
    async with factory.openai() as client:
        backend = SAIAAdapter(client)
        saia = SAIA(backend=backend)
        result = await saia.verify("The sky is blue", "factually accurate")

Requires: pip install llm-infer[saia]
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Self

from llm_saia.core import (
    Backend,
    Message,
    ToolDef,
)
from llm_saia.core import (
    ChatResponse as SAIAChatResponse,
)
from llm_saia.core import (
    ToolCall as SAIAToolCall,
)

if TYPE_CHECKING:
    from .base import ChatClient
    from .types import ChatResponse as InferChatResponse


class SAIAAdapter(Backend):
    """Adapter that wraps a ChatClient to implement llm-saia Backend.

    This allows any ChatClient (LLMClient, LLMRouter, BoundChatClient) to be
    used with the llm-saia verb vocabulary.

    The adapter handles conversion between:
    - SAIA Message types <-> llm-infer message dicts
    - SAIA ToolDef <-> llm-infer tool dicts
    - llm-infer ChatResponse -> SAIA ChatResponse
    - Tool call argument parsing (JSON string -> dict)
    """

    def __init__(self, client: ChatClient) -> None:
        """Initialize the adapter with a ChatClient.

        Args:
            client: The ChatClient to wrap (LLMClient, LLMRouter, etc.).
        """
        self._client = client
        self._chat_args: dict[str, Any] = {}

    def with_chat_args(self, **kwargs: Any) -> Self:
        """Bind kwargs to be merged into every chat call.

        Useful for binding routing parameters when using LLMRouter with
        role-based strategies.

        Args:
            **kwargs: Arguments to merge into chat calls (e.g., role, backend).

        Returns:
            Self for fluent chaining.

        Example:
            adapter = SAIAAdapter(router).with_chat_args(role="exploration")
        """
        self._chat_args = {**self._chat_args, **kwargs}
        return self

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        context: dict[str, Any] | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> SAIAChatResponse:
        """Send a chat completion request via the wrapped LLMClient.

        Args:
            messages: Conversation history in SAIA format.
            system: Optional system prompt.
            tools: Optional tools the LLM can call.
            response_schema: Optional JSON schema for structured output.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (default 1.0).
            context: User context passed to callbacks (cost tracking, tracing).
            abort_signal: Optional event that, when set, aborts the request.
                Raises PauseRequested on abort. Uses streaming internally
                for fast abort even during time-to-first-token.

        Returns:
            SAIA ChatResponse with content, tool calls, token usage, resolved
            model name, and the raw llm-infer response attached as ``raw``.
        """
        api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools) if tools else None
        response_format = self._build_response_format(response_schema)

        call_kwargs: dict[str, Any] = {**self._chat_args, "messages": api_messages}
        if system is not None:
            call_kwargs["system"] = system
        if api_tools is not None:
            call_kwargs["tools"] = api_tools
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        elif "temperature" not in call_kwargs:
            call_kwargs["temperature"] = 1.0
        if context is not None:
            call_kwargs["context"] = context

        if abort_signal is not None:
            return await self._chat_with_abort(call_kwargs, abort_signal)

        response = await self._client.chat_async(**call_kwargs)
        return self._convert_response(response)

    async def _chat_with_abort(
        self,
        call_kwargs: dict[str, Any],
        abort_signal: asyncio.Event,
    ) -> SAIAChatResponse:
        """Stream with abort support via task cancellation."""
        from llm_saia.core.errors import PauseRequested

        from .types import ChatStream

        stream: ChatStream = self._client.chat_stream_async(**call_kwargs)
        stream_task = asyncio.create_task(self._consume_stream(stream, abort_signal))
        abort_task = asyncio.create_task(abort_signal.wait())
        done, pending = await asyncio.wait(
            [stream_task, abort_task], return_when=asyncio.FIRST_COMPLETED
        )
        await self._cancel_tasks(pending)

        if stream_task in done and stream_task.exception() is None:
            if stream.response is not None:
                return self._convert_response(stream.response)
            if abort_signal.is_set():
                raise PauseRequested()
        if abort_task in done:
            raise PauseRequested()
        if stream_task.exception():
            raise stream_task.exception()  # type: ignore[misc]
        raise RuntimeError("No response available after streaming")

    @staticmethod
    async def _consume_stream(stream: Any, abort_signal: asyncio.Event) -> None:
        """Consume stream tokens, returning early if abort fires."""
        async for _ in stream:
            if abort_signal.is_set():
                return

    @staticmethod
    async def _cancel_tasks(tasks: set[asyncio.Task[Any]]) -> None:
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert SAIA messages to llm-infer message dicts."""
        return [self._convert_message(msg) for msg in messages]

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """Convert a single SAIA message to llm-infer format."""
        # OpenAI uses "tool" for tool results, Anthropic convention uses "tool_result".
        # SAIA's Role.TOOL is "tool", so we accept both for compatibility.
        if msg.role in ("tool", "tool_result"):
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content or "",
            }

        if msg.role == "assistant" and msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }

        return {"role": msg.role, "content": msg.content or ""}

    def _convert_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        """Convert SAIA tool definitions to llm-infer format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _build_response_format(
        self, response_schema: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Build response_format dict for structured output."""
        if not response_schema:
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.get("name", "response"),
                "strict": True,
                "schema": response_schema.get("schema", response_schema),
            },
        }

    def _convert_response(self, response: InferChatResponse) -> SAIAChatResponse:
        """Convert llm-infer ChatResponse to SAIA ChatResponse.

        The full llm-infer response is attached as ``raw`` so consumers that
        need backend-specific fields (thinking, adapter info, detailed usage)
        can reach them without another round of adapter churn.
        """
        tool_calls: list[SAIAToolCall] = []

        if response.tool_calls:
            for tc in response.tool_calls:
                arguments = self._parse_tool_arguments(tc.function.arguments)
                tool_calls.append(
                    SAIAToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        finish_reason = response.finish_reason.value if response.finish_reason else None

        return SAIAChatResponse(
            content=response.content or "",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=response.model,
            raw=response,
        )

    def _parse_tool_arguments(self, args_str: str) -> dict[str, Any]:
        """Parse tool arguments from JSON string to dict."""
        try:
            result: dict[str, Any] = json.loads(args_str)
            return result
        except json.JSONDecodeError:
            return {"_error": "malformed_json", "_raw": args_str[:200]}
