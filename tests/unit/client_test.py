"""Unit tests for OpenAI-compatible client."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_infer.api import ChatCompletionUsage, FinishReason
from llm_infer.client import (
    ChatClient,
    ChatResponse,
    OpenAIClient,
    _parse_finish_reason,
    _parse_sse_stream,
    _parse_usage,
)

pytestmark = pytest.mark.unit


class TestChatResponse:
    """Test ChatResponse dataclass."""

    def test_creates_with_content(self) -> None:
        """Test response is created with content."""
        resp = ChatResponse(content="Hello")
        assert resp.content == "Hello"
        assert resp.usage is None
        assert resp.finish_reason is None

    def test_creates_with_all_fields(self) -> None:
        """Test response with all fields populated."""
        usage = ChatCompletionUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        resp = ChatResponse(
            content="Hello",
            usage=usage,
            finish_reason=FinishReason.STOP,
        )
        assert resp.content == "Hello"
        assert resp.usage.total_tokens == 30
        assert resp.finish_reason == FinishReason.STOP


class TestChatClientProtocol:
    """Test ChatClient protocol."""

    def test_openai_client_implements_protocol(self) -> None:
        """Test that OpenAIClient satisfies ChatClient protocol."""
        client = OpenAIClient()
        assert isinstance(client, ChatClient)

    def test_mock_client_implements_protocol(self) -> None:
        """Test that a mock class can implement the protocol."""

        class MockClient:
            def __init__(self) -> None:
                self._last_response: ChatResponse | None = None

            @property
            def last_response(self) -> ChatResponse | None:
                return self._last_response

            async def chat(
                self, messages: list[dict[str, Any]], **kwargs: Any
            ) -> ChatResponse:
                self._last_response = ChatResponse(content="mock")
                return self._last_response

            async def chat_stream(
                self, messages: list[dict[str, Any]], **kwargs: Any
            ) -> AsyncIterator[str]:
                yield "mock"
                self._last_response = ChatResponse(content="mock")

        mock = MockClient()
        assert isinstance(mock, ChatClient)


class TestOpenAIClientInit:
    """Test OpenAIClient initialization."""

    def test_default_values(self) -> None:
        """Test client initializes with defaults."""
        client = OpenAIClient()
        assert client._base_url == "http://localhost:8000/v1"
        assert client._api_key is None
        assert client._timeout == 120.0
        assert client.last_response is None

    def test_custom_values(self) -> None:
        """Test client initializes with custom values."""
        client = OpenAIClient(
            base_url="http://custom:9000/api",
            api_key="sk-test",
            timeout=60.0,
        )
        assert client._base_url == "http://custom:9000/api"
        assert client._api_key == "sk-test"
        assert client._timeout == 60.0

    def test_strips_trailing_slash(self) -> None:
        """Test base_url strips trailing slash."""
        client = OpenAIClient(base_url="http://localhost:8000/v1/")
        assert client._base_url == "http://localhost:8000/v1"


class TestOpenAIClientBuildHelpers:
    """Test OpenAIClient helper methods."""

    def test_build_headers_without_api_key(self) -> None:
        """Test headers without API key."""
        client = OpenAIClient()
        headers = client._build_headers()
        assert headers == {"Content-Type": "application/json"}

    def test_build_headers_with_api_key(self) -> None:
        """Test headers include auth when API key set."""
        client = OpenAIClient(api_key="sk-test")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer sk-test"

    def test_build_messages_with_system(self) -> None:
        """Test messages prepends system prompt."""
        client = OpenAIClient()
        messages = [{"role": "user", "content": "Hello"}]
        result = client._build_messages(messages, system="You are helpful.")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_build_messages_without_system(self) -> None:
        """Test messages without system prompt."""
        client = OpenAIClient()
        messages = [{"role": "user", "content": "Hello"}]
        result = client._build_messages(messages, system=None)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_build_payload_minimal(self) -> None:
        """Test minimal payload construction."""
        client = OpenAIClient()
        messages = [{"role": "user", "content": "Hi"}]
        payload = client._build_payload(
            messages, model="test", temperature=0.7, max_tokens=None, stream=False
        )
        assert payload == {
            "model": "test",
            "messages": messages,
            "temperature": 0.7,
            "stream": False,
        }

    def test_build_payload_with_max_tokens(self) -> None:
        """Test payload includes max_tokens when set."""
        client = OpenAIClient()
        payload = client._build_payload(
            [], model="test", temperature=1.0, max_tokens=100, stream=True
        )
        assert payload["max_tokens"] == 100
        assert payload["stream"] is True

    def test_build_payload_with_kwargs(self) -> None:
        """Test payload includes additional kwargs."""
        client = OpenAIClient()
        payload = client._build_payload(
            [],
            model="test",
            temperature=1.0,
            max_tokens=None,
            stream=False,
            top_p=0.9,
            presence_penalty=0.5,
        )
        assert payload["top_p"] == 0.9
        assert payload["presence_penalty"] == 0.5


class TestParseHelpers:
    """Test parsing helper functions."""

    def test_parse_finish_reason_stop(self) -> None:
        """Test parsing stop finish reason."""
        assert _parse_finish_reason("stop") == FinishReason.STOP

    def test_parse_finish_reason_length(self) -> None:
        """Test parsing length finish reason."""
        assert _parse_finish_reason("length") == FinishReason.LENGTH

    def test_parse_finish_reason_content_filter(self) -> None:
        """Test parsing content_filter finish reason."""
        assert _parse_finish_reason("content_filter") == FinishReason.CONTENT_FILTER

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

    def test_parse_usage_missing_fields(self) -> None:
        """Test parsing usage with missing fields defaults to 0."""
        usage = _parse_usage({})
        assert usage is not None
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestParseSSEStream:
    """Test SSE stream parsing."""

    @pytest.mark.asyncio
    async def test_parses_data_lines(self) -> None:
        """Test parsing standard SSE data lines."""
        mock_response = AsyncMock()
        mock_response.aiter_lines = lambda: _async_iter(
            [
                'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                'data: {"choices": [{"delta": {"content": " World"}}]}',
                "data: [DONE]",
            ]
        )

        chunks = [chunk async for chunk in _parse_sse_stream(mock_response)]
        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[1]["choices"][0]["delta"]["content"] == " World"

    @pytest.mark.asyncio
    async def test_skips_empty_lines(self) -> None:
        """Test that empty lines are skipped."""
        mock_response = AsyncMock()
        mock_response.aiter_lines = lambda: _async_iter(
            [
                "",
                'data: {"test": 1}',
                "",
                "data: [DONE]",
            ]
        )

        chunks = [chunk async for chunk in _parse_sse_stream(mock_response)]
        assert len(chunks) == 1
        assert chunks[0]["test"] == 1

    @pytest.mark.asyncio
    async def test_skips_non_data_lines(self) -> None:
        """Test that non-data lines are skipped."""
        mock_response = AsyncMock()
        mock_response.aiter_lines = lambda: _async_iter(
            [
                ": comment",
                "event: message",
                'data: {"test": 1}',
                "data: [DONE]",
            ]
        )

        chunks = [chunk async for chunk in _parse_sse_stream(mock_response)]
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stops_at_done(self) -> None:
        """Test that parsing stops at [DONE]."""
        mock_response = AsyncMock()
        mock_response.aiter_lines = lambda: _async_iter(
            [
                'data: {"test": 1}',
                "data: [DONE]",
                'data: {"test": 2}',  # Should not be reached
            ]
        )

        chunks = [chunk async for chunk in _parse_sse_stream(mock_response)]
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_skips_malformed_json(self) -> None:
        """Test that malformed JSON is skipped."""
        mock_response = AsyncMock()
        mock_response.aiter_lines = lambda: _async_iter(
            [
                'data: {"valid": 1}',
                "data: {malformed",
                'data: {"valid": 2}',
                "data: [DONE]",
            ]
        )

        chunks = [chunk async for chunk in _parse_sse_stream(mock_response)]
        assert len(chunks) == 2
        assert chunks[0]["valid"] == 1
        assert chunks[1]["valid"] == 2


class TestOpenAIClientChat:
    """Test OpenAIClient.chat method."""

    @pytest.mark.asyncio
    async def test_chat_success(self) -> None:
        """Test successful non-streaming chat request."""
        client = OpenAIClient(base_url="http://test:8000/v1")

        mock_response = MagicMock()
        mock_response.json.return_value = {
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

        with patch("llm_infer.client.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            response = await client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
            )

        assert response.content == "Hello there!"
        assert response.finish_reason == FinishReason.STOP
        assert response.usage is not None
        assert response.usage.total_tokens == 8
        assert client.last_response == response

    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self) -> None:
        """Test chat prepends system prompt."""
        client = OpenAIClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_infer.client.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await client.chat(
                messages=[{"role": "user", "content": "Hi"}],
                system="Be helpful.",
            )

            # Verify payload included system message
            call_args = mock_client.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["messages"][0] == {
                "role": "system",
                "content": "Be helpful.",
            }

    @pytest.mark.asyncio
    async def test_chat_empty_choices_raises(self) -> None:
        """Test that empty choices array raises ValueError."""
        client = OpenAIClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()

        with patch("llm_infer.client.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(ValueError, match="empty choices"):
                await client.chat(messages=[{"role": "user", "content": "Hi"}])


class TestOpenAIClientChatStream:
    """Test OpenAIClient.chat_stream method."""

    @pytest.mark.asyncio
    async def test_chat_stream_yields_tokens(self) -> None:
        """Test streaming yields tokens correctly."""
        client = OpenAIClient()

        sse_lines = [
            'data: {"choices": [{"delta": {"role": "assistant"}}]}',
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " World"}}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
            "data: [DONE]",
        ]

        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_lines = lambda: _async_iter(sse_lines)

        with patch("llm_infer.client.httpx.AsyncClient") as mock_client_class:
            # Setup: AsyncClient() returns client_instance
            # async with client_instance -> enters context, returns same instance
            # client_instance.stream() returns stream_cm
            # async with stream_cm -> enters context, returns response
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            # Make AsyncClient an async context manager
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)

            # Make stream() return an async context manager
            mock_stream_cm = MagicMock()
            mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_response)
            mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
            mock_client_instance.stream.return_value = mock_stream_cm

            tokens = []
            async for token in client.chat_stream(
                messages=[{"role": "user", "content": "Hi"}]
            ):
                tokens.append(token)

        assert tokens == ["Hello", " World"]
        assert client.last_response is not None
        assert client.last_response.content == "Hello World"
        assert client.last_response.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_chat_stream_captures_usage(self) -> None:
        """Test streaming captures usage from final chunk."""
        client = OpenAIClient()

        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hi"}}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}',
            "data: [DONE]",
        ]

        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_lines = lambda: _async_iter(sse_lines)

        with patch("llm_infer.client.httpx.AsyncClient") as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance

            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)

            mock_stream_cm = MagicMock()
            mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_response)
            mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
            mock_client_instance.stream.return_value = mock_stream_cm

            tokens = [t async for t in client.chat_stream(messages=[])]

        assert tokens == ["Hi"]
        assert client.last_response.usage is not None
        assert client.last_response.usage.total_tokens == 6


# Helper for creating async iterators in tests
async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    """Create an async iterator from a list."""
    for item in items:
        yield item
