"""Type definitions for the LLM client.

This module defines the response types returned by the client, including
llm-infer specific extensions like thinking content and tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer.schemas.openai import (
    ChatCompletionUsage,
    FinishReason,
    ToolCall,
)


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

    llm-infer Extensions:
        thinking: Extracted thinking/reasoning content from <think> blocks.
            Only present when think mode is enabled.
        tool_calls: List of tool/function calls made by the model. Present
            when the model invokes tools during generation.
    """

    content: str
    usage: ChatCompletionUsage | None = None
    finish_reason: FinishReason | None = None
    model: str | None = None
    # llm-infer extensions
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = field(default=None)

    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0
