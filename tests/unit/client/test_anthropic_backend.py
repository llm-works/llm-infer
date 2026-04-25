"""Unit tests for Anthropic backend."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from appinfra.log import Logger

from llm_infer.client import ChatRequest

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger for testing."""
    return MagicMock(spec=Logger)


class TestAnthropicBackendImport:
    """Test Anthropic backend import behavior."""

    def test_import_error_when_package_missing(self, mock_lg: Logger) -> None:
        """Test ImportError raised when anthropic package not installed."""
        with patch.dict("sys.modules", {"anthropic": None}):
            import importlib

            import llm_infer.client.backends.anthropic as anthropic_backend

            importlib.reload(anthropic_backend)

            with pytest.raises(ImportError, match="anthropic"):
                anthropic_backend.AnthropicBackend(mock_lg, "test")


class TestAnthropicBackendMocked:
    """Test Anthropic backend with mocked SDK."""

    @pytest.fixture
    def mock_anthropic(self) -> Any:
        """Create mock anthropic module."""
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_async_client = MagicMock()

        mock_module.Anthropic.return_value = mock_client
        mock_module.AsyncAnthropic.return_value = mock_async_client

        mock_module.APIConnectionError = type(
            "APIConnectionError", (Exception,), {"message": "connection error"}
        )
        mock_module.APITimeoutError = type(
            "APITimeoutError", (Exception,), {"message": "timeout"}
        )
        mock_module.APIStatusError = type(
            "APIStatusError",
            (Exception,),
            {"message": "status error", "status_code": 400},
        )

        return mock_module

    def test_convert_messages_filters_system(self, mock_anthropic: Any) -> None:
        """Test system messages are filtered from message list."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            result = backend._convert_messages(messages)

            assert len(result) == 1
            assert result[0]["role"] == "user"

    def test_convert_messages_handles_tool_response(self, mock_anthropic: Any) -> None:
        """Test tool response messages are converted to tool_result format."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            messages = [
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "The weather is sunny",
                },
            ]
            result = backend._convert_messages(messages)

            assert len(result) == 1
            assert result[0]["role"] == "user"
            assert result[0]["content"][0]["type"] == "tool_result"
            assert result[0]["content"][0]["tool_use_id"] == "call_123"

    def test_convert_messages_merges_consecutive_tool_results(
        self, mock_anthropic: Any
    ) -> None:
        """Test consecutive tool results are merged into single user message."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            messages = [
                {"role": "user", "content": "Search for both"},
                {
                    "role": "assistant",
                    "content": "Searching...",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "function": {"name": "search", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Result 1"},
                {"role": "tool", "tool_call_id": "call_2", "content": "Result 2"},
            ]
            result = backend._convert_messages(messages)

            assert len(result) == 3
            tool_result_msg = result[2]
            assert tool_result_msg["role"] == "user"
            assert len(tool_result_msg["content"]) == 2
            assert tool_result_msg["content"][0]["tool_use_id"] == "call_1"
            assert tool_result_msg["content"][1]["tool_use_id"] == "call_2"

    def test_convert_assistant_with_tools_parses_arguments(
        self, mock_anthropic: Any
    ) -> None:
        """Test tool call arguments are parsed from JSON strings."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "function": {
                            "name": "get_temp",
                            "arguments": "{bad json",
                        },
                    },
                ],
            }
            converted = backend._convert_single_message(msg)

            assert converted is not None
            tool_uses = [b for b in converted["content"] if b["type"] == "tool_use"]
            assert tool_uses[0]["input"] == {"city": "SF"}
            assert tool_uses[1]["input"] == {}

    def test_prepare_request_basic(self, mock_anthropic: Any) -> None:
        """Test basic request kwargs construction."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._default_model = "default-model"
            backend._max_tokens = 4096

            request = ChatRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-opus",
                temperature=0.7,
                max_tokens=1000,
                system="Be helpful",
            )
            kwargs = backend._prepare_request(request)

            assert kwargs["model"] == "claude-3-opus"
            assert kwargs["max_tokens"] == 1000
            assert kwargs["temperature"] == 0.7
            assert kwargs["system"] == "Be helpful"

    def test_prepare_request_converts_tools(self, mock_anthropic: Any) -> None:
        """Test tools are converted to Anthropic format."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._default_model = "default-model"
            backend._max_tokens = 4096

            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

            request = ChatRequest(
                messages=[],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                tools=openai_tools,
            )
            kwargs = backend._prepare_request(request)

            assert "tools" in kwargs
            assert len(kwargs["tools"]) == 1
            assert kwargs["tools"][0]["name"] == "get_weather"
            assert kwargs["tools"][0]["description"] == "Get weather for a city"

    def test_prepare_request_tool_choice_auto(self, mock_anthropic: Any) -> None:
        """Test tool_choice auto is converted."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._default_model = "default-model"
            backend._max_tokens = 4096

            request = ChatRequest(
                messages=[],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                tool_choice="auto",
            )
            kwargs = backend._prepare_request(request)

            assert kwargs["tool_choice"] == {"type": "auto"}

    def test_prepare_request_tool_choice_required(self, mock_anthropic: Any) -> None:
        """Test tool_choice required maps to any."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._default_model = "default-model"
            backend._max_tokens = 4096

            request = ChatRequest(
                messages=[],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                tool_choice="required",
            )
            kwargs = backend._prepare_request(request)

            assert kwargs["tool_choice"] == {"type": "any"}

    def test_map_stop_reason(self, mock_anthropic: Any) -> None:
        """Test stop reason mapping."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend
            from llm_infer.schemas.openai import FinishReason

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            assert backend._map_stop_reason("end_turn") == FinishReason.STOP
            assert backend._map_stop_reason("stop_sequence") == FinishReason.STOP
            assert backend._map_stop_reason("max_tokens") == FinishReason.LENGTH
            assert backend._map_stop_reason("tool_use") == FinishReason.TOOL_CALLS
            assert backend._map_stop_reason(None) is None
            assert backend._map_stop_reason("unknown") is None


class TestAnthropicStructuredOutput:
    """Test structured output (response_format) handling."""

    @pytest.fixture
    def mock_anthropic(self) -> Any:
        """Create mock anthropic module."""
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.Anthropic.return_value = mock_client
        mock_module.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_module.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_module.APIStatusError = type("APIStatusError", (Exception,), {})
        return mock_module

    def test_convert_response_format_none(self, mock_anthropic: Any) -> None:
        """Test None response_format returns None."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            result = backend._convert_response_format_to_tool(None)
            assert result is None

    def test_convert_response_format_text(self, mock_anthropic: Any) -> None:
        """Test text type response_format returns None."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            result = backend._convert_response_format_to_tool({"type": "text"})
            assert result is None

    def test_convert_response_format_json_object(self, mock_anthropic: Any) -> None:
        """Test json_object type creates basic schema tool."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            result = backend._convert_response_format_to_tool({"type": "json_object"})

            assert result is not None
            assert result["name"] == "__structured_output__"
            assert result["input_schema"] == {"type": "object"}

    def test_convert_response_format_json_schema(self, mock_anthropic: Any) -> None:
        """Test json_schema type creates tool with full schema."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema},
            }

            result = backend._convert_response_format_to_tool(response_format)

            assert result is not None
            assert result["name"] == "__structured_output__"
            assert result["input_schema"] == schema

    def test_prepare_request_with_response_format(self, mock_anthropic: Any) -> None:
        """Test response_format adds tool and tool_choice to request."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._default_model = "default-model"
            backend._max_tokens = 4096

            request = ChatRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                extra={"response_format": {"type": "json_object"}},
            )
            kwargs = backend._prepare_request(request)

            assert "tools" in kwargs
            assert len(kwargs["tools"]) == 1
            assert kwargs["tools"][0]["name"] == "__structured_output__"

            assert kwargs["tool_choice"] == {
                "type": "tool",
                "name": "__structured_output__",
            }

            assert kwargs["_structured_output_tool"] == "__structured_output__"

    def test_prepare_request_strips_response_format(self, mock_anthropic: Any) -> None:
        """Test response_format is not passed to API."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._default_model = "default-model"
            backend._max_tokens = 4096

            request = ChatRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                extra={"response_format": {"type": "json_object"}},
            )
            kwargs = backend._prepare_request(request)

            assert "response_format" not in kwargs

    def test_prepare_request_merges_with_existing_tools(
        self, mock_anthropic: Any
    ) -> None:
        """Test response_format tool is merged with user-provided tools."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._default_model = "default-model"
            backend._max_tokens = 4096

            user_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ]

            request = ChatRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                tools=user_tools,
                extra={"response_format": {"type": "json_object"}},
            )
            kwargs = backend._prepare_request(request)

            assert len(kwargs["tools"]) == 2
            assert kwargs["tools"][0]["name"] == "get_weather"
            assert kwargs["tools"][1]["name"] == "__structured_output__"

    def test_parse_response_structured_output(self, mock_anthropic: Any) -> None:
        """Test structured output tool args extracted as content."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            mock_block = MagicMock()
            mock_block.type = "tool_use"
            mock_block.name = "__structured_output__"
            mock_block.input = {"name": "Alice", "age": 30}

            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.stop_reason = "tool_use"
            mock_response.model = "claude-3"
            mock_response.usage = None

            result = backend._parse_response(
                mock_response,
                "claude-3",
                structured_output_tool="__structured_output__",
            )

            assert result.content == '{"name": "Alice", "age": 30}'
            assert result.tool_calls is None

    def test_structured_output_finish_reason(self, mock_anthropic: Any) -> None:
        """Test structured output returns STOP not TOOL_CALLS."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend
            from llm_infer.schemas.openai import FinishReason

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            mock_block = MagicMock()
            mock_block.type = "tool_use"
            mock_block.name = "__structured_output__"
            mock_block.input = {"result": "test"}

            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.stop_reason = "tool_use"
            mock_response.model = "claude-3"
            mock_response.usage = None

            result = backend._parse_response(
                mock_response,
                "claude-3",
                structured_output_tool="__structured_output__",
            )

            assert result.finish_reason == FinishReason.STOP

    def test_parse_response_regular_tool_call_unaffected(
        self, mock_anthropic: Any
    ) -> None:
        """Test regular tool calls still work when structured output is enabled."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic

            mock_block = MagicMock()
            mock_block.type = "tool_use"
            mock_block.name = "get_weather"
            mock_block.id = "call_123"
            mock_block.input = {"city": "NYC"}

            mock_response = MagicMock()
            mock_response.content = [mock_block]
            mock_response.stop_reason = "tool_use"
            mock_response.model = "claude-3"
            mock_response.usage = None

            result = backend._parse_response(mock_response, "claude-3", None)

            assert result.tool_calls is not None
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].function.name == "get_weather"
            assert result.content == ""

    def test_prepare_request_extracts_response_format_from_extra(
        self, mock_anthropic: Any
    ) -> None:
        """Test response_format in extra is extracted and handled."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._default_model = "default-model"
            backend._max_tokens = 4096

            request = ChatRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                extra={"response_format": {"type": "json_object"}},
            )
            kwargs = backend._prepare_request(request)

            assert "tools" in kwargs
            assert len(kwargs["tools"]) == 1
            assert kwargs["tools"][0]["name"] == "__structured_output__"

            assert kwargs["tool_choice"] == {
                "type": "tool",
                "name": "__structured_output__",
            }

            assert "response_format" not in kwargs
