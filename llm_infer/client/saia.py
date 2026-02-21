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

import json
from typing import TYPE_CHECKING, Any

from llm_saia.core.backend import (
    AgentResponse,
    Backend,
    Message,
    ToolDef,
)
from llm_saia.core.backend import (
    ToolCall as SAIAToolCall,
)

if TYPE_CHECKING:
    from llm_infer.client.client import LLMClient


class SAIAAdapter(Backend):
    """Adapter that wraps LLMClient to implement llm-saia Backend.

    This allows any LLMClient (or subclass like LearnClient) to be used
    with the llm-saia verb vocabulary.

    The adapter handles conversion between:
    - SAIA Message types <-> llm-infer message dicts
    - SAIA ToolDef <-> llm-infer tool dicts
    - llm-infer ChatResponse -> SAIA AgentResponse
    - Tool call argument parsing (JSON string -> dict)
    """

    def __init__(self, client: LLMClient) -> None:
        """Initialize the adapter with an LLMClient.

        Args:
            client: The LLMClient instance to wrap. Can be any subclass.
        """
        self._client = client

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AgentResponse:
        """Send a chat completion request via the wrapped LLMClient.

        Args:
            messages: Conversation history in SAIA format.
            system: Optional system prompt.
            tools: Optional tools the LLM can call.
            response_schema: Optional JSON schema for structured output.
            max_tokens: Maximum tokens to generate.

        Returns:
            AgentResponse with content, tool calls, and token usage.
        """
        api_messages = self._convert_messages(messages)
        api_tools = self._convert_tools(tools) if tools else None
        response_format = self._build_response_format(response_schema)

        response = await self._client.chat_async(
            messages=api_messages,
            system=system,
            tools=api_tools,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        return self._convert_response(response)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert SAIA messages to llm-infer message dicts."""
        return [self._convert_message(msg) for msg in messages]

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        """Convert a single SAIA message to llm-infer format."""
        if msg.role == "tool_result":
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

    def _convert_response(self, response: Any) -> AgentResponse:
        """Convert llm-infer ChatResponse to SAIA AgentResponse."""
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

        return AgentResponse(
            content=response.content or "",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _parse_tool_arguments(self, args_str: str) -> dict[str, Any]:
        """Parse tool arguments from JSON string to dict."""
        try:
            result: dict[str, Any] = json.loads(args_str)
            return result
        except json.JSONDecodeError:
            return {"_error": "malformed_json", "_raw": args_str[:200]}
