"""Unit tests for Anthropic backend."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestAnthropicBackendImport:
    """Test Anthropic backend import behavior."""

    def test_import_error_when_package_missing(self) -> None:
        """Test ImportError raised when anthropic package not installed."""
        with patch.dict("sys.modules", {"anthropic": None}):
            # Need to reimport to trigger the check
            import importlib

            import llm_infer.client.backends.anthropic as anthropic_backend

            # Force reimport
            importlib.reload(anthropic_backend)

            # The import itself won't fail, but instantiation will
            # This is because we do lazy import in __init__
            with pytest.raises(ImportError, match="anthropic"):
                anthropic_backend.AnthropicBackend()


class TestAnthropicBackendMocked:
    """Test Anthropic backend with mocked SDK.

    These tests mock the anthropic package to test the backend logic
    without requiring the actual package to be installed.
    """

    @pytest.fixture
    def mock_anthropic(self) -> Any:
        """Create mock anthropic module."""
        mock_module = MagicMock()

        # Create mock client classes
        mock_client = MagicMock()
        mock_async_client = MagicMock()

        mock_module.Anthropic.return_value = mock_client
        mock_module.AsyncAnthropic.return_value = mock_async_client

        # Create exception classes
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

            # System message should be filtered
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
            assert tool_uses[1]["input"] == {}  # Malformed JSON defaults to empty dict

    def test_prepare_request_basic(self, mock_anthropic: Any) -> None:
        """Test basic request kwargs construction."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._model = "default-model"
            backend._max_tokens = 4096

            kwargs = backend._prepare_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-opus",
                temperature=0.7,
                max_tokens=1000,
                system="Be helpful",
                think=None,
                tools=None,
                tool_choice=None,
            )

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
            backend._model = "default-model"
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

            kwargs = backend._prepare_request(
                messages=[],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                system=None,
                think=None,
                tools=openai_tools,
                tool_choice=None,
            )

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
            backend._model = "default-model"
            backend._max_tokens = 4096

            kwargs = backend._prepare_request(
                messages=[],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                system=None,
                think=None,
                tools=None,
                tool_choice="auto",
            )

            assert kwargs["tool_choice"] == {"type": "auto"}

    def test_prepare_request_tool_choice_required(self, mock_anthropic: Any) -> None:
        """Test tool_choice required maps to any."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            backend = AnthropicBackend.__new__(AnthropicBackend)
            backend._anthropic = mock_anthropic
            backend._model = "default-model"
            backend._max_tokens = 4096

            kwargs = backend._prepare_request(
                messages=[],
                model="claude-3",
                temperature=1.0,
                max_tokens=100,
                system=None,
                think=None,
                tools=None,
                tool_choice="required",
            )

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

    def test_from_config(self, mock_anthropic: Any) -> None:
        """Test creating backend from config."""
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from llm_infer.client.backends.anthropic import AnthropicBackend

            config = {
                "model": "claude-3-opus",
                "api_key": "test-key",
                "max_tokens": 2000,
                "timeout": 60.0,
            }

            backend = AnthropicBackend.from_config(config)

            assert backend._model == "claude-3-opus"
            assert backend._max_tokens == 2000
            assert backend._timeout == 60.0
