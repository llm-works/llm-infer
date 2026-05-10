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
    ChatRequest,
)
from llm_infer.client.backends.providers.openai import (
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
        backend = OpenAICompatibleBackend(mock_lg, "test")
        assert backend._base_url == "http://localhost:8000/v1"
        assert backend.default_model is None
        assert backend._api_key is None
        assert backend._ctx.request_timeout == 120.0
        assert backend.last_response is None
        backend.close()

    def test_custom_values(self, mock_lg: Logger) -> None:
        """Test backend initializes with custom values."""
        from llm_infer.client.backends import BackendContext

        ctx = BackendContext(request_timeout=60.0)
        backend = OpenAICompatibleBackend(
            mock_lg,
            "test",
            ctx=ctx,
            default_model="gpt-4",
            base_url="http://custom:9000/api",
            api_key="sk-test",
        )
        assert backend._base_url == "http://custom:9000/api"
        assert backend.default_model == "gpt-4"
        assert backend._api_key == "sk-test"
        assert backend._ctx.request_timeout == 60.0
        backend.close()

    def test_strips_trailing_slash(self, mock_lg: Logger) -> None:
        """Test base_url strips trailing slash."""
        backend = OpenAICompatibleBackend(
            mock_lg, "test", base_url="http://localhost:8000/v1/"
        )
        assert backend._base_url == "http://localhost:8000/v1"
        backend.close()


class TestOpenAICompatibleBackendHelpers:
    """Test OpenAICompatibleBackend helper methods."""

    def test_build_headers_without_api_key(self, mock_lg: Logger) -> None:
        """Test headers without API key."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        headers = backend._build_headers()
        assert headers == {"Content-Type": "application/json"}
        backend.close()

    def test_build_headers_with_api_key(self, mock_lg: Logger) -> None:
        """Test headers include auth when API key set."""
        backend = OpenAICompatibleBackend(mock_lg, "test", api_key="sk-test")
        headers = backend._build_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        backend.close()

    def test_build_messages_with_system(self, mock_lg: Logger) -> None:
        """Test messages prepends system prompt."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        messages = [{"role": "user", "content": "Hello"}]
        result = backend._build_messages(messages, system="You are helpful.")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hello"}
        backend.close()

    def test_build_messages_without_system(self, mock_lg: Logger) -> None:
        """Test messages without system prompt."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        messages = [{"role": "user", "content": "Hello"}]
        result = backend._build_messages(messages, system=None)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}
        backend.close()

    def test_build_payload_minimal(self, mock_lg: Logger) -> None:
        """Test minimal payload construction."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(messages=messages, model="test-model", temperature=0.7)
        payload = backend._build_payload(request, messages, stream=False)
        assert payload == {
            "model": "test-model",
            "messages": messages,
            "temperature": 0.7,
            "stream": False,
        }
        backend.close()

    def test_build_payload_with_llm_infer_extensions(self, mock_lg: Logger) -> None:
        """Test payload includes llm-infer extensions."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        request = ChatRequest(
            messages=[],
            model="test-model",
            temperature=1.0,
            max_tokens=100,
            adapter="my-lora",
            think=True,
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
        )
        payload = backend._build_payload(request, [], stream=True)
        assert payload["max_tokens"] == 100
        assert payload["adapter"] == "my-lora"
        assert payload["think"] is True
        assert payload["tools"] is not None
        assert payload["tool_choice"] == "auto"
        backend.close()

    def test_build_payload_extracts_extra_contents(self, mock_lg: Logger) -> None:
        """Test extra contents are merged as top-level keys."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            messages=messages,
            model="test-model",
            temperature=1.0,
            extra={"response_format": {"type": "json_object"}, "custom_param": 42},
        )
        payload = backend._build_payload(request, messages, stream=False)
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["custom_param"] == 42
        backend.close()

    def test_build_payload_extra_respects_reserved_keys(self, mock_lg: Logger) -> None:
        """Test extra cannot override reserved keys like model."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            messages=messages,
            model="original-model",
            temperature=1.0,
            extra={
                "model": "evil-override",
                "response_format": {"type": "json_object"},
            },
        )
        payload = backend._build_payload(request, messages, stream=False)
        # Reserved key should not be overridden
        assert payload["model"] == "original-model"
        # Non-reserved key should be added
        assert payload["response_format"] == {"type": "json_object"}
        backend.close()

    def test_build_payload_extra_filters_none_values(self, mock_lg: Logger) -> None:
        """Test extra filters out None values."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            messages=messages,
            model="test-model",
            temperature=1.0,
            extra={
                "response_format": {"type": "json_object"},
                "custom_param": None,
            },
        )
        payload = backend._build_payload(request, messages, stream=False)
        assert payload["response_format"] == {"type": "json_object"}
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
        backend = OpenAICompatibleBackend(
            mock_lg, "test", base_url="http://test:8000/v1"
        )

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

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4",
            temperature=0.7,
        )
        with patch.object(backend._client, "post", return_value=mock_response):
            response = backend.chat(request)

        assert response.content == "Hello there!"
        assert response.finish_reason == FinishReason.STOP
        assert response.usage is not None
        assert response.usage.total_tokens == 8
        assert response.model == "gpt-4"
        assert backend.last_response == response
        backend.close()

    def test_chat_with_tool_calls(self, mock_lg: Logger) -> None:
        """Test chat returns tool calls when present."""
        backend = OpenAICompatibleBackend(mock_lg, "test")

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

        request = ChatRequest(messages=[{"role": "user", "content": "Weather?"}])
        with patch.object(backend._client, "post", return_value=mock_response):
            response = backend.chat(request)

        assert response.has_tool_calls()
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.tool_calls[0].function.name == "get_weather"
        assert response.finish_reason == FinishReason.TOOL_CALLS
        backend.close()

    def test_chat_connect_error_raises_unavailable(self, mock_lg: Logger) -> None:
        """Test connection error raises BackendUnavailableError."""
        backend = OpenAICompatibleBackend(mock_lg, "test")

        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])
        with patch.object(
            backend._client, "post", side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(BackendUnavailableError, match="Failed to connect"):
                backend.chat(request)

        backend.close()

    def test_chat_timeout_error_raises_timeout(self, mock_lg: Logger) -> None:
        """Test timeout error raises BackendTimeoutError."""
        backend = OpenAICompatibleBackend(mock_lg, "test")

        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])
        with patch.object(
            backend._client, "post", side_effect=httpx.TimeoutException("timeout")
        ):
            with pytest.raises(BackendTimeoutError, match="timed out"):
                backend.chat(request)

        backend.close()

    def test_chat_http_error_raises_request_error(self, mock_lg: Logger) -> None:
        """Test HTTP error raises BackendRequestError."""
        backend = OpenAICompatibleBackend(mock_lg, "test")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        http_error = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )

        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])
        with patch.object(backend._client, "post", side_effect=http_error):
            with pytest.raises(BackendRequestError) as exc_info:
                backend.chat(request)

        assert exc_info.value.status_code == 400
        backend.close()


class TestOpenAICompatibleBackendChatAsync:
    """Test OpenAICompatibleBackend async methods."""

    @pytest.mark.asyncio
    async def test_chat_async_success(self, mock_lg: Logger) -> None:
        """Test successful async chat request."""
        backend = OpenAICompatibleBackend(mock_lg, "test")

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
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])
        with patch.object(
            async_client, "post", new=AsyncMock(return_value=mock_response)
        ):
            response = await backend.chat_async(request)

        assert response.content == "Hello!"
        assert response.finish_reason == FinishReason.STOP
        await backend.aclose()


class TestOpenAICompatibleBackendStream:
    """Test OpenAICompatibleBackend streaming methods."""

    def test_chat_stream_yields_tokens(self, mock_lg: Logger) -> None:
        """Test streaming yields tokens correctly."""
        backend = OpenAICompatibleBackend(mock_lg, "test")

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

        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])
        with patch.object(backend._client, "stream", return_value=mock_stream):
            tokens = list(backend.chat_stream(request))

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
        with OpenAICompatibleBackend(mock_lg, "test") as backend:
            assert backend._client is not None

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_clients(self, mock_lg: Logger) -> None:
        """Test async context manager closes both clients."""
        async with OpenAICompatibleBackend(mock_lg, "test") as backend:
            _ = backend._get_async_client()
            assert backend._async_client is not None

        assert backend._async_client is None

    @pytest.mark.asyncio
    async def test_lazy_async_client_creation(self, mock_lg: Logger) -> None:
        """Test async client is created lazily."""
        backend = OpenAICompatibleBackend(mock_lg, "test")
        assert backend._async_client is None

        client = backend._get_async_client()
        assert client is not None
        assert backend._async_client is client

        client2 = backend._get_async_client()
        assert client2 is client

        await backend.aclose()


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    """Create an async iterator from a list."""
    for item in items:
        yield item


class TestOpenAICompatibleBackendConcurrentClose:
    """Test concurrent request handling during close."""

    @pytest.mark.asyncio
    async def test_aclose_waits_for_active_requests(self, mock_lg: Logger) -> None:
        """Test aclose() waits for in-flight requests to complete."""
        import asyncio

        backend = OpenAICompatibleBackend(mock_lg, "test")
        request_started = asyncio.Event()
        allow_response = asyncio.Event()

        async def slow_post(*args: object, **kwargs: object) -> MagicMock:
            request_started.set()
            await allow_response.wait()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "model": "test",
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            }
            mock_response.raise_for_status = MagicMock()
            return mock_response

        async_client = backend._get_async_client()
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        with patch.object(async_client, "post", new=slow_post):
            # Start async request
            chat_task = asyncio.create_task(backend.chat_async(request))
            await request_started.wait()

            # Start close while request is in flight
            close_task = asyncio.create_task(backend.aclose())

            # Close should be waiting (active request count > 0)
            await asyncio.sleep(0.01)
            assert not close_task.done()
            assert backend._active_async_requests == 1

            # Allow the request to complete
            allow_response.set()
            response = await chat_task

            # Now close should complete
            await close_task

        assert response.content == "done"
        assert backend._async_client is None

    @pytest.mark.asyncio
    async def test_new_requests_fail_during_close(self, mock_lg: Logger) -> None:
        """Test new requests fail immediately after close is requested."""
        backend = OpenAICompatibleBackend(mock_lg, "test")

        # Request close
        backend._close_requested = True

        # New request should fail
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])
        with pytest.raises(BackendUnavailableError, match="Backend is closing"):
            await backend.chat_async(request)

        # Clean up
        backend._close_requested = False
        await backend.aclose()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, mock_lg: Logger) -> None:
        """Test multiple concurrent requests complete successfully."""
        import asyncio

        backend = OpenAICompatibleBackend(mock_lg, "test")

        async def mock_post(*args: object, **kwargs: object) -> MagicMock:
            await asyncio.sleep(0.01)  # Small delay to ensure concurrency
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "model": "test",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            }
            mock_response.raise_for_status = MagicMock()
            return mock_response

        async_client = backend._get_async_client()
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        with patch.object(async_client, "post", new=mock_post):
            # Start multiple concurrent requests
            tasks = [asyncio.create_task(backend.chat_async(request)) for _ in range(5)]

            # Verify active count increases
            await asyncio.sleep(0.001)
            assert backend._active_async_requests > 0

            # Wait for all to complete
            responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(r.content == "ok" for r in responses)
        assert backend._active_async_requests == 0

        await backend.aclose()

    @pytest.mark.asyncio
    async def test_concurrent_close_with_multiple_requests(
        self, mock_lg: Logger
    ) -> None:
        """Test aclose() waits for all concurrent requests."""
        import asyncio

        backend = OpenAICompatibleBackend(mock_lg, "test")
        requests_started = 0
        allow_responses = asyncio.Event()

        async def slow_post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal requests_started
            requests_started += 1
            await allow_responses.wait()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "model": "test",
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            }
            mock_response.raise_for_status = MagicMock()
            return mock_response

        async_client = backend._get_async_client()
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        with patch.object(async_client, "post", new=slow_post):
            # Start multiple concurrent requests
            tasks = [asyncio.create_task(backend.chat_async(request)) for _ in range(3)]

            # Wait for all requests to start
            while requests_started < 3:
                await asyncio.sleep(0.001)

            # Start close
            close_task = asyncio.create_task(backend.aclose())
            await asyncio.sleep(0.01)

            # Close should be waiting
            assert not close_task.done()
            assert backend._active_async_requests == 3

            # Allow responses
            allow_responses.set()

            # All tasks should complete
            responses = await asyncio.gather(*tasks)
            await close_task

        assert len(responses) == 3
        assert backend._async_client is None
