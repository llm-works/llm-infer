"""Unit tests for OpenAI-compatible tool/function calling support."""

import pytest

from llm_infer.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    FinishReason,
    FunctionCall,
    FunctionDefinition,
    Role,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolChoice,
    ToolChoiceFunction,
    ToolChoiceObject,
)
from llm_infer.serving.api.openai.mappers import (
    determine_finish_reason,
    tool_choice_to_dict,
    tools_to_dict,
)
from llm_infer.serving.api.openai.router import _convert_tool_calls
from llm_infer.serving.api.openai.streaming import _convert_tool_calls_to_deltas
from llm_infer.serving.dispatch.types import Request, Response, StreamChunk

pytestmark = pytest.mark.unit


# =============================================================================
# Schema Tests
# =============================================================================


class TestToolSchemas:
    """Test Pydantic schemas for tool calling."""

    def test_function_definition_minimal(self) -> None:
        """Test FunctionDefinition with minimal fields."""
        func = FunctionDefinition(name="get_weather")
        assert func.name == "get_weather"
        assert func.description is None
        assert func.parameters is None

    def test_function_definition_full(self) -> None:
        """Test FunctionDefinition with all fields."""
        func = FunctionDefinition(
            name="get_weather",
            description="Get the current weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        assert func.name == "get_weather"
        assert func.description == "Get the current weather"
        assert func.parameters["type"] == "object"

    def test_tool_definition(self) -> None:
        """Test Tool wrapper around FunctionDefinition."""
        tool = Tool(
            type="function",
            function=FunctionDefinition(name="search", description="Search the web"),
        )
        assert tool.type == "function"
        assert tool.function.name == "search"

    def test_tool_call(self) -> None:
        """Test ToolCall in response."""
        tc = ToolCall(
            id="call_abc123",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"location": "NYC"}'),
        )
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"location": "NYC"}'

    def test_tool_call_delta_streaming(self) -> None:
        """Test ToolCallDelta for streaming responses."""
        delta = ToolCallDelta(
            index=0,
            id="call_xyz789",
            type="function",
            function=FunctionCall(name="calculate", arguments="{}"),
        )
        assert delta.index == 0
        assert delta.id == "call_xyz789"

    def test_tool_choice_string_literals(self) -> None:
        """Test tool_choice string values."""
        # These should be valid ToolChoice values
        choices: list[ToolChoice] = ["auto", "none", "required"]
        for choice in choices:
            assert choice in ("auto", "none", "required")

    def test_tool_choice_object(self) -> None:
        """Test tool_choice as specific function."""
        choice = ToolChoiceObject(
            type="function", function=ToolChoiceFunction(name="get_weather")
        )
        assert choice.type == "function"
        assert choice.function.name == "get_weather"

    def test_chat_message_with_tool_calls(self) -> None:
        """Test ChatMessage containing tool_calls (assistant response)."""
        msg = ChatMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="fn1", arguments="{}"),
                )
            ],
        )
        assert msg.role == Role.ASSISTANT
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_chat_message_tool_response(self) -> None:
        """Test ChatMessage as tool response."""
        msg = ChatMessage(
            role=Role.TOOL,
            content='{"result": 42}',
            tool_call_id="call_1",
        )
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call_1"

    def test_finish_reason_tool_calls(self) -> None:
        """Test TOOL_CALLS finish reason exists."""
        assert FinishReason.TOOL_CALLS.value == "tool_calls"


class TestChatCompletionRequestWithTools:
    """Test ChatCompletionRequest with tool fields."""

    def test_request_with_tools(self) -> None:
        """Test request containing tools list."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=Role.USER, content="What's the weather?")],
            tools=[
                Tool(
                    type="function",
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get weather for location",
                        parameters={"type": "object", "properties": {}},
                    ),
                )
            ],
            tool_choice="auto",
        )
        assert len(request.tools) == 1
        assert request.tool_choice == "auto"

    def test_request_with_specific_tool_choice(self) -> None:
        """Test request with specific function tool_choice."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=Role.USER, content="Calculate 2+2")],
            tools=[
                Tool(
                    type="function",
                    function=FunctionDefinition(name="calculate"),
                )
            ],
            tool_choice=ToolChoiceObject(
                type="function", function=ToolChoiceFunction(name="calculate")
            ),
        )
        assert isinstance(request.tool_choice, ToolChoiceObject)
        assert request.tool_choice.function.name == "calculate"


# =============================================================================
# Mapper Tests
# =============================================================================


class TestToolsToDict:
    """Test tools_to_dict conversion."""

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert tools_to_dict(None) is None

    def test_empty_list_returns_empty(self) -> None:
        """Test empty list returns empty list."""
        assert tools_to_dict([]) == []

    def test_single_tool_conversion(self) -> None:
        """Test single tool converts to dict."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="test_fn", description="A test function"
                ),
            )
        ]
        result = tools_to_dict(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_fn"
        assert result[0]["function"]["description"] == "A test function"

    def test_excludes_none_values(self) -> None:
        """Test None fields are excluded from output."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(name="minimal"),
            )
        ]
        result = tools_to_dict(tools)
        # description and parameters should not be present
        assert "description" not in result[0]["function"]
        assert "parameters" not in result[0]["function"]


class TestToolChoiceToDict:
    """Test tool_choice_to_dict conversion."""

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert tool_choice_to_dict(None) is None

    def test_string_passthrough(self) -> None:
        """Test string values pass through unchanged."""
        assert tool_choice_to_dict("auto") == "auto"
        assert tool_choice_to_dict("none") == "none"
        assert tool_choice_to_dict("required") == "required"

    def test_object_converts_to_dict(self) -> None:
        """Test ToolChoiceObject converts to dict."""
        choice = ToolChoiceObject(
            type="function", function=ToolChoiceFunction(name="my_function")
        )
        result = tool_choice_to_dict(choice)
        assert result["type"] == "function"
        assert result["function"]["name"] == "my_function"


class TestDetermineFinishReason:
    """Test determine_finish_reason with tool_calls parameter."""

    def test_tool_calls_takes_precedence(self) -> None:
        """Test has_tool_calls=True returns TOOL_CALLS."""
        result = determine_finish_reason(
            is_eos=True, max_tokens_reached=False, has_tool_calls=True
        )
        assert result == FinishReason.TOOL_CALLS

    def test_tool_calls_over_max_tokens(self) -> None:
        """Test tool_calls takes precedence over max_tokens."""
        result = determine_finish_reason(
            is_eos=False, max_tokens_reached=True, has_tool_calls=True
        )
        assert result == FinishReason.TOOL_CALLS

    def test_content_filter_highest_precedence(self) -> None:
        """Test guard_triggered still takes highest precedence."""
        result = determine_finish_reason(
            is_eos=True,
            max_tokens_reached=False,
            guard_triggered=True,
            has_tool_calls=True,
        )
        assert result == FinishReason.CONTENT_FILTER

    def test_no_tool_calls_normal_behavior(self) -> None:
        """Test normal finish reason when no tool calls."""
        result = determine_finish_reason(
            is_eos=True, max_tokens_reached=False, has_tool_calls=False
        )
        assert result == FinishReason.STOP

        result = determine_finish_reason(
            is_eos=False, max_tokens_reached=True, has_tool_calls=False
        )
        assert result == FinishReason.LENGTH


# =============================================================================
# Router Conversion Tests
# =============================================================================


class TestConvertToolCalls:
    """Test _convert_tool_calls in router."""

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert _convert_tool_calls(None) is None

    def test_empty_list_returns_none(self) -> None:
        """Test empty list returns None."""
        assert _convert_tool_calls([]) is None

    def test_basic_conversion(self) -> None:
        """Test basic Ollama format to OpenAI format conversion."""
        tool_calls = [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "London"}',
                }
            }
        ]
        result = _convert_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        assert result[0].function.arguments == '{"location": "London"}'
        # ID should be generated with proper format
        assert result[0].id.startswith("call_")
        assert len(result[0].id) == 29  # "call_" + 24 chars

    def test_preserves_existing_id(self) -> None:
        """Test existing ID is preserved."""
        tool_calls = [
            {
                "id": "existing_call_id",
                "function": {"name": "test", "arguments": "{}"},
            }
        ]
        result = _convert_tool_calls(tool_calls)
        assert result[0].id == "existing_call_id"

    def test_multiple_tool_calls(self) -> None:
        """Test multiple tool calls are converted."""
        tool_calls = [
            {"function": {"name": "fn1", "arguments": "{}"}},
            {"function": {"name": "fn2", "arguments": '{"x": 1}'}},
        ]
        result = _convert_tool_calls(tool_calls)
        assert len(result) == 2
        assert result[0].function.name == "fn1"
        assert result[1].function.name == "fn2"
        # Each should have unique ID
        assert result[0].id != result[1].id

    def test_skips_malformed_tool_call_missing_name(self) -> None:
        """Test malformed tool call with missing name is skipped."""
        tool_calls = [
            {"function": {"arguments": "{}"}},  # Missing name
            {"function": {"name": "valid", "arguments": "{}"}},
        ]
        result = _convert_tool_calls(tool_calls)
        # Only the valid one should be returned
        assert len(result) == 1
        assert result[0].function.name == "valid"

    def test_skips_empty_name(self) -> None:
        """Test tool call with empty name is skipped."""
        tool_calls = [
            {"function": {"name": "", "arguments": "{}"}},
            {"function": {"name": "valid", "arguments": "{}"}},
        ]
        result = _convert_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0].function.name == "valid"

    def test_all_malformed_returns_none(self) -> None:
        """Test all malformed tool calls returns None."""
        tool_calls = [
            {"function": {"name": "", "arguments": "{}"}},
            {"function": {"arguments": "{}"}},
        ]
        result = _convert_tool_calls(tool_calls)
        assert result is None

    def test_default_arguments(self) -> None:
        """Test missing arguments defaults to '{}'."""
        tool_calls = [{"function": {"name": "test"}}]
        result = _convert_tool_calls(tool_calls)
        assert result[0].function.arguments == "{}"

    def test_missing_function_dict(self) -> None:
        """Test missing function dict is handled."""
        tool_calls = [{}]  # No function key
        result = _convert_tool_calls(tool_calls)
        # Should be skipped due to missing name
        assert result is None


# =============================================================================
# Streaming Conversion Tests
# =============================================================================


class TestConvertToolCallsToDeltas:
    """Test _convert_tool_calls_to_deltas in streaming."""

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        assert _convert_tool_calls_to_deltas(None) is None

    def test_empty_list_returns_none(self) -> None:
        """Test empty list returns None."""
        assert _convert_tool_calls_to_deltas([]) is None

    def test_basic_conversion(self) -> None:
        """Test basic conversion to delta format."""
        tool_calls = [
            {
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}',
                }
            }
        ]
        result = _convert_tool_calls_to_deltas(tool_calls)
        assert len(result) == 1
        assert isinstance(result[0], ToolCallDelta)
        assert result[0].index == 0
        assert result[0].function.name == "search"
        # ID should have proper format
        assert result[0].id.startswith("call_")
        assert len(result[0].id) == 29

    def test_index_increments(self) -> None:
        """Test index increments for multiple tool calls."""
        tool_calls = [
            {"function": {"name": "fn1", "arguments": "{}"}},
            {"function": {"name": "fn2", "arguments": "{}"}},
            {"function": {"name": "fn3", "arguments": "{}"}},
        ]
        result = _convert_tool_calls_to_deltas(tool_calls)
        assert result[0].index == 0
        assert result[1].index == 1
        assert result[2].index == 2

    def test_preserves_existing_id(self) -> None:
        """Test existing ID is preserved in streaming."""
        tool_calls = [
            {
                "id": "call_preserved",
                "function": {"name": "test", "arguments": "{}"},
            }
        ]
        result = _convert_tool_calls_to_deltas(tool_calls)
        assert result[0].id == "call_preserved"

    def test_skips_malformed_missing_name(self) -> None:
        """Test malformed tool call with missing name is skipped."""
        tool_calls = [
            {"function": {"arguments": "{}"}},
            {"function": {"name": "valid", "arguments": "{}"}},
        ]
        result = _convert_tool_calls_to_deltas(tool_calls)
        assert len(result) == 1
        assert result[0].function.name == "valid"
        # Index is sequential for valid tool calls (0-based)
        assert result[0].index == 0

    def test_all_malformed_returns_none(self) -> None:
        """Test all malformed returns None."""
        tool_calls = [{"function": {"name": ""}}]
        result = _convert_tool_calls_to_deltas(tool_calls)
        assert result is None

    def test_type_always_function(self) -> None:
        """Test type is always 'function'."""
        tool_calls = [{"function": {"name": "test", "arguments": "{}"}}]
        result = _convert_tool_calls_to_deltas(tool_calls)
        assert result[0].type == "function"


# =============================================================================
# Dispatch Types Tests
# =============================================================================


class TestDispatchTypesToolCalling:
    """Test tool calling fields in dispatch types."""

    def test_request_with_tools(self) -> None:
        """Test Request dataclass accepts tool fields."""
        req = Request(
            id="req-1",
            prompt="test prompt",
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
        )
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"

    def test_request_tool_choice_object(self) -> None:
        """Test Request accepts dict tool_choice."""
        req = Request(
            id="req-1",
            prompt="test",
            tool_choice={"type": "function", "function": {"name": "specific"}},
        )
        assert req.tool_choice["function"]["name"] == "specific"

    def test_response_with_tool_calls(self) -> None:
        """Test Response dataclass accepts tool_calls."""
        resp = Response(
            id="resp-1",
            status="completed",
            result="",
            tool_calls=[{"function": {"name": "test", "arguments": "{}"}}],
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1

    def test_stream_chunk_with_tool_calls(self) -> None:
        """Test StreamChunk accepts tool_calls on final chunk."""
        chunk = StreamChunk(
            id="chunk-1",
            token="",
            is_final=True,
            finish_reason="tool_calls",
            tool_calls=[{"function": {"name": "fn", "arguments": "{}"}}],
        )
        assert chunk.finish_reason == "tool_calls"
        assert chunk.tool_calls is not None


# =============================================================================
# ID Format Consistency Tests
# =============================================================================


class TestToolCallIdFormat:
    """Test tool call ID generation is consistent."""

    def test_router_id_format(self) -> None:
        """Test router generates proper ID format."""
        tool_calls = [{"function": {"name": "test", "arguments": "{}"}}]
        result = _convert_tool_calls(tool_calls)
        id_val = result[0].id
        # Should be "call_" + 24 hex chars
        assert id_val.startswith("call_")
        suffix = id_val[5:]
        assert len(suffix) == 24
        # Should be valid hex
        int(suffix, 16)

    def test_streaming_id_format(self) -> None:
        """Test streaming generates proper ID format."""
        tool_calls = [{"function": {"name": "test", "arguments": "{}"}}]
        result = _convert_tool_calls_to_deltas(tool_calls)
        id_val = result[0].id
        # Should be "call_" + 24 hex chars
        assert id_val.startswith("call_")
        suffix = id_val[5:]
        assert len(suffix) == 24
        # Should be valid hex
        int(suffix, 16)

    def test_unique_ids_router(self) -> None:
        """Test router generates unique IDs for each call."""
        tool_calls = [
            {"function": {"name": "fn1", "arguments": "{}"}},
            {"function": {"name": "fn2", "arguments": "{}"}},
        ]
        result = _convert_tool_calls(tool_calls)
        assert result[0].id != result[1].id

    def test_unique_ids_streaming(self) -> None:
        """Test streaming generates unique IDs for each call."""
        tool_calls = [
            {"function": {"name": "fn1", "arguments": "{}"}},
            {"function": {"name": "fn2", "arguments": "{}"}},
        ]
        result = _convert_tool_calls_to_deltas(tool_calls)
        assert result[0].id != result[1].id
