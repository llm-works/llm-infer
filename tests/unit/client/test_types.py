"""Unit tests for client types."""

import pytest

from llm_infer.client import ChatResponse
from llm_infer.schemas.openai import (
    ChatCompletionUsage,
    FinishReason,
    FunctionCall,
    ToolCall,
)

pytestmark = pytest.mark.unit


class TestChatResponse:
    """Test ChatResponse dataclass."""

    def test_creates_with_content_only(self) -> None:
        """Test response is created with just content."""
        resp = ChatResponse(content="Hello")
        assert resp.content == "Hello"
        assert resp.usage is None
        assert resp.finish_reason is None
        assert resp.model is None
        assert resp.thinking is None
        assert resp.tool_calls is None

    def test_creates_with_all_fields(self) -> None:
        """Test response with all fields populated."""
        usage = ChatCompletionUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        tool_calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
            )
        ]
        resp = ChatResponse(
            content="Hello",
            usage=usage,
            finish_reason=FinishReason.STOP,
            model="gpt-4",
            thinking="I should greet the user",
            tool_calls=tool_calls,
        )
        assert resp.content == "Hello"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 30
        assert resp.finish_reason == FinishReason.STOP
        assert resp.model == "gpt-4"
        assert resp.thinking == "I should greet the user"
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "get_weather"

    def test_has_tool_calls_returns_true_when_present(self) -> None:
        """Test has_tool_calls returns True when tool_calls exist."""
        resp = ChatResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="test", arguments="{}"),
                )
            ],
        )
        assert resp.has_tool_calls() is True

    def test_has_tool_calls_returns_false_when_none(self) -> None:
        """Test has_tool_calls returns False when tool_calls is None."""
        resp = ChatResponse(content="Hello")
        assert resp.has_tool_calls() is False

    def test_has_tool_calls_returns_false_when_empty(self) -> None:
        """Test has_tool_calls returns False when tool_calls is empty list."""
        resp = ChatResponse(content="Hello", tool_calls=[])
        assert resp.has_tool_calls() is False

    def test_finish_reason_tool_calls(self) -> None:
        """Test response with tool_calls finish reason."""
        resp = ChatResponse(
            content="",
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="test", arguments="{}"),
                )
            ],
        )
        assert resp.finish_reason == FinishReason.TOOL_CALLS
        assert resp.has_tool_calls()
