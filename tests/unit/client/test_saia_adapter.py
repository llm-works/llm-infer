"""Unit tests for SAIA adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from llm_saia.core import (
    ChatResponse as SAIAChatResponse,
)
from llm_saia.core import (
    Message,
    ToolDef,
)
from llm_saia.core import (
    ToolCall as SAIAToolCall,
)

from llm_infer.client import ChatResponse, LLMClient
from llm_infer.client.saia import SAIAAdapter
from llm_infer.schemas.openai import (
    ChatCompletionUsage,
    FinishReason,
    FunctionCall,
    ToolCall,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock LLMClient."""
    client = MagicMock(spec=LLMClient)
    client.chat_async = AsyncMock()
    return client


class TestSAIAAdapterInit:
    """Test SAIAAdapter initialization."""

    def test_init_with_client(self, mock_client: MagicMock) -> None:
        """Test adapter initializes with client."""
        adapter = SAIAAdapter(mock_client)
        assert adapter._client is mock_client


class TestSAIAAdapterMessageConversion:
    """Test message conversion from SAIA to llm-infer format."""

    def test_convert_user_message(self, mock_client: MagicMock) -> None:
        """Test user message conversion."""
        adapter = SAIAAdapter(mock_client)
        msg = Message(role="user", content="Hello")

        result = adapter._convert_message(msg)

        assert result == {"role": "user", "content": "Hello"}

    def test_convert_assistant_message(self, mock_client: MagicMock) -> None:
        """Test assistant message conversion."""
        adapter = SAIAAdapter(mock_client)
        msg = Message(role="assistant", content="Hi there")

        result = adapter._convert_message(msg)

        assert result == {"role": "assistant", "content": "Hi there"}

    def test_convert_tool_result_message(self, mock_client: MagicMock) -> None:
        """Test tool result message conversion."""
        adapter = SAIAAdapter(mock_client)
        msg = Message(
            role="tool_result",
            content='{"result": 42}',
            tool_call_id="call_123",
        )

        result = adapter._convert_message(msg)

        assert result == {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"result": 42}',
        }

    def test_convert_tool_message(self, mock_client: MagicMock) -> None:
        """Test tool message conversion (SAIA uses role='tool', not 'tool_result')."""
        adapter = SAIAAdapter(mock_client)
        msg = Message(
            role="tool",
            content="Search results...",
            tool_call_id="toolu_01ABC123",
        )

        result = adapter._convert_message(msg)

        assert result == {
            "role": "tool",
            "tool_call_id": "toolu_01ABC123",
            "content": "Search results...",
        }

    def test_convert_assistant_message_with_tool_calls(
        self, mock_client: MagicMock
    ) -> None:
        """Test assistant message with tool calls conversion."""
        adapter = SAIAAdapter(mock_client)
        msg = Message(
            role="assistant",
            content="Let me call a tool",
            tool_calls=[
                SAIAToolCall(
                    id="call_123",
                    name="get_weather",
                    arguments={"city": "NYC"},
                ),
            ],
        )

        result = adapter._convert_message(msg)

        assert result["role"] == "assistant"
        assert result["content"] == "Let me call a tool"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"city": "NYC"}'


class TestSAIAAdapterToolConversion:
    """Test tool definition conversion."""

    def test_convert_tools(self, mock_client: MagicMock) -> None:
        """Test tool definition conversion."""
        adapter = SAIAAdapter(mock_client)
        tools = [
            ToolDef(
                name="get_weather",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ]

        result = adapter._convert_tools(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a city"
        assert result[0]["function"]["parameters"]["type"] == "object"


class TestSAIAAdapterResponseConversion:
    """Test response conversion from llm-infer to SAIA format."""

    def test_convert_simple_response(self, mock_client: MagicMock) -> None:
        """Test simple text response conversion."""
        adapter = SAIAAdapter(mock_client)
        response = ChatResponse(
            content="Hello!",
            usage=ChatCompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
            finish_reason=FinishReason.STOP,
        )

        result = adapter._convert_response(response)

        assert isinstance(result, SAIAChatResponse)
        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_convert_response_with_tool_calls(self, mock_client: MagicMock) -> None:
        """Test response with tool calls conversion."""
        adapter = SAIAAdapter(mock_client)
        response = ChatResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    type="function",
                    function=FunctionCall(
                        name="get_weather",
                        arguments='{"city": "NYC"}',
                    ),
                ),
            ],
            finish_reason=FinishReason.TOOL_CALLS,
        )

        result = adapter._convert_response(response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "NYC"}
        assert result.finish_reason == "tool_calls"

    def test_convert_response_no_usage(self, mock_client: MagicMock) -> None:
        """Test response conversion when usage is None."""
        adapter = SAIAAdapter(mock_client)
        response = ChatResponse(content="Hello!", usage=None)

        result = adapter._convert_response(response)

        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_convert_response_no_finish_reason(self, mock_client: MagicMock) -> None:
        """Test response conversion when finish_reason is None."""
        adapter = SAIAAdapter(mock_client)
        response = ChatResponse(content="Hello!", finish_reason=None)

        result = adapter._convert_response(response)

        assert result.finish_reason is None

    def test_convert_response_passes_model_and_raw(
        self, mock_client: MagicMock
    ) -> None:
        """Resolved model name and raw llm-infer response are passed through.

        Motivation: ``model: auto`` in config resolves server-side. Without the
        passthrough, SAIA consumers can't attribute cost to the actual model,
        and backend-specific fields (thinking, adapter info) are unreachable.
        """
        adapter = SAIAAdapter(mock_client)
        response = ChatResponse(
            content="Hello!",
            finish_reason=FinishReason.STOP,
            model="claude-haiku-4-5-20251001",
        )

        result = adapter._convert_response(response)

        assert result.model == "claude-haiku-4-5-20251001"
        assert result.raw is response
        # raw is a live reference, not a copy: mutations are visible post-conversion.
        response.content = "mutated"
        assert result.raw.content == "mutated"


class TestSAIAAdapterToolArgumentParsing:
    """Test tool argument parsing."""

    def test_parse_valid_json(self, mock_client: MagicMock) -> None:
        """Test parsing valid JSON arguments."""
        adapter = SAIAAdapter(mock_client)

        result = adapter._parse_tool_arguments('{"city": "NYC", "units": "celsius"}')

        assert result == {"city": "NYC", "units": "celsius"}

    def test_parse_invalid_json(self, mock_client: MagicMock) -> None:
        """Test parsing invalid JSON returns error dict."""
        adapter = SAIAAdapter(mock_client)

        result = adapter._parse_tool_arguments("not valid json")

        assert result["_error"] == "malformed_json"
        assert "_raw" in result


class TestSAIAAdapterResponseFormat:
    """Test response format building."""

    def test_build_response_format_none(self, mock_client: MagicMock) -> None:
        """Test None schema returns None."""
        adapter = SAIAAdapter(mock_client)

        result = adapter._build_response_format(None)

        assert result is None

    def test_build_response_format_with_schema(self, mock_client: MagicMock) -> None:
        """Test schema is converted to response_format."""
        adapter = SAIAAdapter(mock_client)
        schema = {
            "name": "Person",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }

        result = adapter._build_response_format(schema)

        assert result is not None
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "Person"
        assert result["json_schema"]["strict"] is True

    def test_build_response_format_default_name(self, mock_client: MagicMock) -> None:
        """Test schema without name uses default."""
        adapter = SAIAAdapter(mock_client)
        schema = {"schema": {"type": "object"}}

        result = adapter._build_response_format(schema)

        assert result["json_schema"]["name"] == "response"


class TestSAIAAdapterChat:
    """Test the main chat method."""

    @pytest.mark.asyncio
    async def test_chat_simple(self, mock_client: MagicMock) -> None:
        """Test simple chat call."""
        mock_client.chat_async.return_value = ChatResponse(
            content="Hello!",
            usage=ChatCompletionUsage(
                prompt_tokens=5, completion_tokens=2, total_tokens=7
            ),
            finish_reason=FinishReason.STOP,
        )
        adapter = SAIAAdapter(mock_client)

        result = await adapter.chat(
            messages=[Message(role="user", content="Hi")],
            system="You are helpful",
            max_tokens=100,
        )

        assert isinstance(result, SAIAChatResponse)
        assert result.content == "Hello!"
        mock_client.chat_async.assert_called_once()
        call_kwargs = mock_client.chat_async.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, mock_client: MagicMock) -> None:
        """Test chat with tool definitions."""
        mock_client.chat_async.return_value = ChatResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="search", arguments='{"q": "test"}'),
                ),
            ],
            finish_reason=FinishReason.TOOL_CALLS,
        )
        adapter = SAIAAdapter(mock_client)
        tools = [
            ToolDef(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        ]

        result = await adapter.chat(
            messages=[Message(role="user", content="Search for test")],
            tools=tools,
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        call_kwargs = mock_client.chat_async.call_args.kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 1

    @pytest.mark.asyncio
    async def test_chat_with_response_schema(self, mock_client: MagicMock) -> None:
        """Test chat with structured output schema."""
        mock_client.chat_async.return_value = ChatResponse(
            content='{"name": "Alice"}',
            finish_reason=FinishReason.STOP,
        )
        adapter = SAIAAdapter(mock_client)
        schema = {
            "name": "Person",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
        }

        await adapter.chat(
            messages=[Message(role="user", content="Create a person")],
            response_schema=schema,
        )

        call_kwargs = mock_client.chat_async.call_args.kwargs
        assert call_kwargs["response_format"] is not None
        assert call_kwargs["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, mock_client: MagicMock) -> None:
        """Test chat passes temperature to client."""
        mock_client.chat_async.return_value = ChatResponse(
            content="Hello!",
            finish_reason=FinishReason.STOP,
        )
        adapter = SAIAAdapter(mock_client)

        await adapter.chat(
            messages=[Message(role="user", content="Hi")],
            temperature=0.7,
        )

        call_kwargs = mock_client.chat_async.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_temperature_default(self, mock_client: MagicMock) -> None:
        """Test chat uses default temperature 1.0 when not specified."""
        mock_client.chat_async.return_value = ChatResponse(
            content="Hello!",
            finish_reason=FinishReason.STOP,
        )
        adapter = SAIAAdapter(mock_client)

        await adapter.chat(messages=[Message(role="user", content="Hi")])

        call_kwargs = mock_client.chat_async.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0
