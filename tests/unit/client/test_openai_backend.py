"""Unit tests for OpenAI-compatible backend."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from appinfra.log import Logger

from llm_infer.client import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_infer.client.backends.openai import (
    OpenAICompatibleBackend,
    _parse_finish_reason,
    _parse_usage,
)
from llm_infer.schemas.openai import FinishReason

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger for testing."""
    return MagicMock(spec=Logger)


class TestOpenAICompatibleBackendInit:
    """Test OpenAICompatibleBackend initialization."""

    def test_default_values(self, mock_lg: Logger) -> None:
        """Test backend initializes with defaults."""
        backend = OpenAICompatibleBackend(mock_lg)
        assert backend._base_url == "http://localhost:8000/v1"
        assert backend._model == "default"
        assert backend._api_key is None
        assert backend._timeout == 120.0
        assert backend.last_response is None
        backend.close()

    def test_custom_values(self, mock_lg: Logger) -> None:
        """Test backend initializes with custom values."""
        backend = OpenAICompatibleBackend(
            mock_lg,
            base_url="http://custom:9000/api",
            model="gpt-4",
            api_key="sk-test",
            timeout=60.0,
        )
        assert backend._base_url == "http://custom:9000/api"
        assert backend._model == "gpt-4"
        assert backend._api_key == "sk-test"
        assert backend._timeout == 60.0
        backend.close()

    def test_strips_trailing_slash(self, mock_lg: Logger) -> None:
        """Test base_url strips trailing slash."""
        backend = OpenAICompatibleBackend(mock_lg, base_url="http://localhost:8000/v1/")
        assert backend._base_url == "http://localhost:8000/v1"
        backend.close()

    def test_from_config(self, mock_lg: Logger) -> None:
        """Test creating backend from config dict."""
        config = {
            "base_url": "http://test:8000/v1",
            "model": "test-model",
            "api_key": "test-key",
            "timeout": 30.0,
        }
        backend = OpenAICompatibleBackend.from_config(mock_lg, config)
        assert backend._base_url == "http://test:8000/v1"
        assert backend._model == "test-model"
        assert backend._api_key == "test-key"
        assert backend._timeout == 30.0
        backend.close()


class TestOpenAICompatibleBackendHelpers:
    """Test OpenAICompatibleBackend helper methods."""

    def test_build_headers_without_api_key(self, mock_lg: Logger) -> None:
        """Test headers without API key."""
        backend = OpenAICompatibleBackend(mock_lg)
        headers = backend._build_headers()
        assert headers == {"Content-Type": "application/json"}
        backend.close()

    def test_build_headers_with_api_key(self, mock_lg: Logger) -> None:
        """Test headers include auth when API key set."""
        backend = OpenAICompatibleBackend(mock_lg, api_key="sk-test")
        headers = backend._build_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        backend.close()

    def test_build_messages_with_system(self, mock_lg: Logger) -> None:
        """Test messages prepends system prompt."""
        backend = OpenAICompatibleBackend(mock_lg)
        messages = [{"role": "user", "content": "Hello"}]
        result = backend._build_messages(messages, system="You are helpful.")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hello"}
        backend.close()

    def test_build_messages_without_system(self, mock_lg: Logger) -> None:
        """Test messages without system prompt."""
        backend = OpenAICompatibleBackend(mock_lg)
        messages = [{"role": "user", "content": "Hello"}]
        result = backend._build_messages(messages, system=None)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}
        backend.close()

    def test_build_payload_minimal(self, mock_lg: Logger) -> None:
        """Test minimal payload construction."""
        backend = OpenAICompatibleBackend(mock_lg)
        messages = [{"role": "user", "content": "Hi"}]
        payload = backend._build_payload(
            messages=messages,
            model="test",
            temperature=0.7,
            max_tokens=None,
            stream=False,
            adapter=None,
            think=None,
            tools=None,
            tool_choice=None,
        )
        assert payload == {
            "model": "test",
            "messages": messages,
            "temperature": 0.7,
            "stream": False,
        }
        backend.close()

    def test_build_payload_with_llm_infer_extensions(self, mock_lg: Logger) -> None:
        """Test payload includes llm-infer extensions."""
        backend = OpenAICompatibleBackend(mock_lg)
        payload = backend._build_payload(
            messages=[],
            model="test",
            temperature=1.0,
            max_tokens=100,
            stream=True,
            adapter="my-lora",
            think=True,
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
        )
        assert payload["max_tokens"] == 100
        assert payload["adapter"] == "my-lora"
        assert payload["think"] is True
        assert payload["tools"] is not None
        assert payload["tool_choice"] == "auto"
        backend.close()

    def test_build_payload_extracts_extra_body_contents(self, mock_lg: Logger) -> None:
        """Test extra_body contents are merged as top-level keys."""
        backend = OpenAICompatibleBackend(mock_lg)
        payload = backend._build_payload(
            messages=[{"role": "user", "content": "Hi"}],
            model="test",
            temperature=1.0,
            max_tokens=None,
            stream=False,
            adapter=None,
            think=None,
            tools=None,
            tool_choice=None,
            extra_body={"response_format": {"type": "json_object"}, "custom_param": 42},
        )
        # extra_body contents should be top-level, not nested
        assert "extra_body" not in payload
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["custom_param"] == 42
        backend.close()

    def test_build_payload_extra_body_respects_reserved_keys(
        self, mock_lg: Logger
    ) -> None:
        """Test extra_body cannot override reserved keys like model."""
        backend = OpenAICompatibleBackend(mock_lg)
        payload = backend._build_payload(
            messages=[{"role": "user", "content": "Hi"}],
            model="original-model",
            temperature=1.0,
            max_tokens=None,
            stream=False,
            adapter=None,
            think=None,
            tools=None,
            tool_choice=None,
            extra_body={
                "model": "evil-override",
                "response_format": {"type": "json_object"},
            },
        )
        # Reserved key should not be overridden
        assert payload["model"] == "original-model"
        # Non-reserved key should be added
        assert payload["response_format"] == {"type": "json_object"}
        backend.close()

    def test_build_payload_extra_body_filters_none_values(
        self, mock_lg: Logger
    ) -> None:
        """Test extra_body filters out None values (consistent with kwargs behavior)."""
        backend = OpenAICompatibleBackend(mock_lg)
        payload = backend._build_payload(
            messages=[{"role": "user", "content": "Hi"}],
            model="test",
            temperature=1.0,
            max_tokens=None,
            stream=False,
            adapter=None,
            think=None,
            tools=None,
            tool_choice=None,
            extra_body={
                "response_format": {"type": "json_object"},
                "custom_param": None,  # Should be filtered out
            },
        )
        # Non-None value should be added
        assert payload["response_format"] == {"type": "json_object"}
        # None value should be filtered out
        assert "custom_param" not in payload
        backend.close()


class TestParseHelpers:
    """Test parsing helper functions."""

    def test_parse_finish_reason_stop(self) -> None:
        """Test parsing stop finish reason."""
        assert _parse_finish_reason("stop") == FinishReason.STOP

    def test_parse_finish_reason_length(self) -> None:
        """Test parsing length finish reason."""
        assert _parse_finish_reason("length") == FinishReason.LENGTH

    def test_parse_finish_reason_tool_calls(self) -> None:
        """Test parsing tool_calls finish reason."""
        assert _parse_finish_reason("tool_calls") == FinishReason.TOOL_CALLS

    def test_parse_finish_reason_none(self) -> None:
        """Test parsing None finish reason."""
        assert _parse_finish_reason(None) is None

    def test_parse_finish_reason_unknown(self) -> None:
        """Test parsing unknown finish reason returns None."""
        assert _parse_finish_reason("unknown_reason") is None

    def test_parse_usage(self) -> None:
        """Test parsing usage dict."""
        data = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        usage = _parse_usage(data)
        assert usage is not None
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_parse_usage_none(self) -> None:
        """Test parsing None usage."""
        assert _parse_usage(None) is None


class TestOpenAICompatibleBackendChat:
    """Test OpenAICompatibleBackend.chat method."""

    def test_chat_success(self, mock_lg: Logger) -> None:
        """Test successful non-streaming chat request."""
        backend = OpenAICompatibleBackend(mock_lg, base_url="http://test:8000/v1")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "gpt-4",
            "choices": [
                {
                    "message": {"content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(backend._client, "post", return_value=mock_response):
            response = backend.chat(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
            )

        assert response.content == "Hello there!"
        assert response.finish_reason == FinishReason.STOP
        assert response.usage is not None
        assert response.usage.total_tokens == 8
        assert response.model == "gpt-4"
        assert backend.last_response == response
        backend.close()

    def test_chat_with_tool_calls(self, mock_lg: Logger) -> None:
        """Test chat returns tool calls when present."""
        backend = OpenAICompatibleBackend(mock_lg)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(backend._client, "post", return_value=mock_response):
            response = backend.chat(messages=[{"role": "user", "content": "Weather?"}])

        assert response.has_tool_calls()
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].function.name == "get_weather"
        assert response.finish_reason == FinishReason.TOOL_CALLS
        backend.close()

    def test_chat_connect_error_raises_unavailable(self, mock_lg: Logger) -> None:
        """Test connection error raises BackendUnavailableError."""
        backend = OpenAICompatibleBackend(mock_lg)

        with patch.object(
            backend._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(BackendUnavailableError, match="Failed to connect"):
                backend.chat(messages=[{"role": "user", "content": "Hi"}])

        backend.close()

    def test_chat_timeout_error_raises_timeout(self, mock_lg: Logger) -> None:
        """Test timeout error raises BackendTimeoutError."""
        backend = OpenAICompatibleBackend(mock_lg)

        with patch.object(
            backend._client, "post", side_effect=httpx.TimeoutException("timeout")
        ):
            with pytest.raises(BackendTimeoutError, match="timed out"):
                backend.chat(messages=[{"role": "user", "content": "Hi"}])

        backend.close()

    def test_chat_http_error_raises_request_error(self, mock_lg: Logger) -> None:
        """Test HTTP error raises BackendRequestError."""
        backend = OpenAICompatibleBackend(mock_lg)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        http_error = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )

        with patch.object(backend._client, "post", side_effect=http_error):
            with pytest.raises(BackendRequestError) as exc_info:
                backend.chat(messages=[{"role": "user", "content": "Hi"}])

        assert exc_info.value.status_code == 400
        backend.close()


class TestOpenAICompatibleBackendChatAsync:
    """Test OpenAICompatibleBackend async methods."""

    @pytest.mark.asyncio
    async def test_chat_async_success(self, mock_lg: Logger) -> None:
        """Test successful async chat request."""
        backend = OpenAICompatibleBackend(mock_lg)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "gpt-4",
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_response.raise_for_status = MagicMock()

        async_client = backend._get_async_client()
        with patch.object(
            async_client, "post", new=AsyncMock(return_value=mock_response)
        ):
            response = await backend.chat_async(
                messages=[{"role": "user", "content": "Hi"}]
            )

        assert response.content == "Hello!"
        assert response.finish_reason == FinishReason.STOP
        await backend.aclose()


class TestOpenAICompatibleBackendStream:
    """Test OpenAICompatibleBackend streaming methods."""

    def test_chat_stream_yields_tokens(self, mock_lg: Logger) -> None:
        """Test streaming yields tokens correctly."""
        backend = OpenAICompatibleBackend(mock_lg)

        sse_lines = [
            'data: {"choices": [{"delta": {"role": "assistant"}}]}',
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " World"}}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}',
            "data: [DONE]",
        ]

        mock_stream = MagicMock()
        mock_stream.raise_for_status = MagicMock()
        mock_stream.iter_lines.return_value = iter(sse_lines)
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)

        with patch.object(backend._client, "stream", return_value=mock_stream):
            tokens = list(
                backend.chat_stream(messages=[{"role": "user", "content": "Hi"}])
            )

        assert tokens == ["Hello", " World"]
        assert backend.last_response is not None
        assert backend.last_response.content == "Hello World"
        assert backend.last_response.finish_reason == FinishReason.STOP
        assert backend.last_response.usage is not None
        assert backend.last_response.usage.total_tokens == 7
        backend.close()


class TestOpenAICompatibleBackendResourceManagement:
    """Test resource management."""

    def test_context_manager_closes_client(self, mock_lg: Logger) -> None:
        """Test context manager closes sync client."""
        with OpenAICompatibleBackend(mock_lg) as backend:
            assert backend._client is not None

        # After exit, client should be closed (though we can't easily verify)

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_clients(self, mock_lg: Logger) -> None:
        """Test async context manager closes both clients."""
        async with OpenAICompatibleBackend(mock_lg) as backend:
            # Force async client creation
            _ = backend._get_async_client()
            assert backend._async_client is not None

        # After exit, async client should be None
        assert backend._async_client is None

    @pytest.mark.asyncio
    async def test_lazy_async_client_creation(self, mock_lg: Logger) -> None:
        """Test async client is created lazily."""
        backend = OpenAICompatibleBackend(mock_lg)
        assert backend._async_client is None

        client = backend._get_async_client()
        assert client is not None
        assert backend._async_client is client

        # Second call returns same client
        client2 = backend._get_async_client()
        assert client2 is client

        await backend.aclose()


# Helper for creating async iterators in tests
async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    """Create an async iterator from a list."""
    for item in items:
        yield item
