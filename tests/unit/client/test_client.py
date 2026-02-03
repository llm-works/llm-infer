"""Unit tests for LLMClient facade and Factory."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import ChatResponse, Factory, LLMClient
from llm_infer.client.backends import Backend, OpenAICompatibleBackend
from llm_infer.schemas.openai import ChatCompletionUsage, FinishReason

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger for testing."""
    return MagicMock(spec=Logger)


class MockBackend(Backend):
    """Mock backend for testing."""

    def __init__(
        self, lg: Logger | None = None, responses: list[ChatResponse] | None = None
    ) -> None:
        self._lg = lg
        self._responses = iter(responses or [])
        self._last_response: ChatResponse | None = None
        self._closed = False
        self._aclosed = False

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        response = next(self._responses)
        self._last_response = response
        return response

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
        response = next(self._responses)
        yield from response.content
        self._last_response = response

    async def chat_async(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatResponse:
        response = next(self._responses)
        self._last_response = response
        return response

    async def chat_stream_async(self, messages: list[dict[str, Any]], **kwargs: Any):
        response = next(self._responses)
        for char in response.content:
            yield char
        self._last_response = response

    def close(self) -> None:
        self._closed = True

    async def aclose(self) -> None:
        self._aclosed = True

    @classmethod
    def from_config(cls, lg: Logger, config: dict[str, Any]) -> "MockBackend":
        return cls(lg=lg)


class TestLLMClientInit:
    """Test LLMClient initialization."""

    def test_init_with_backend(self) -> None:
        """Test client initializes with backend."""
        backend = MockBackend()
        client = LLMClient(backend=backend)
        assert client.backend is backend
        assert client.last_response is None

    def test_init_with_default_model(self) -> None:
        """Test client stores default model."""
        backend = MockBackend()
        client = LLMClient(backend=backend, default_model="gpt-4")
        assert client._default_model == "gpt-4"


class TestFactory:
    """Test Factory methods."""

    def test_openai_creates_openai_backend(self, mock_lg: Logger) -> None:
        """Test openai() creates OpenAI backend."""
        factory = Factory(mock_lg)
        client = factory.openai(
            base_url="http://test:8000/v1",
            model="test-model",
            api_key="test-key",
            timeout=30.0,
        )
        assert isinstance(client.backend, OpenAICompatibleBackend)
        assert client._default_model == "test-model"
        client.close()

    def test_from_config_single_backend(self, mock_lg: Logger) -> None:
        """Test from_config with single backend config."""
        factory = Factory(mock_lg)
        config = {
            "type": "openai_compatible",
            "base_url": "http://test:8000/v1",
            "model": "test-model",
        }
        client = factory.from_config(config)
        assert isinstance(client.backend, OpenAICompatibleBackend)
        client.close()

    def test_from_config_multi_backend(self, mock_lg: Logger) -> None:
        """Test from_config with multiple backends."""
        factory = Factory(mock_lg)
        config = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                    "model": "local-model",
                },
                "remote": {
                    "type": "openai_compatible",
                    "base_url": "http://remote:8000/v1",
                    "model": "remote-model",
                },
            },
        }
        client = factory.from_config(config)
        assert isinstance(client.backend, OpenAICompatibleBackend)
        assert client._default_model == "local-model"
        client.close()

    def test_from_config_uses_first_backend_if_no_default(
        self, mock_lg: Logger
    ) -> None:
        """Test from_config uses first backend when no default specified."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "first": {
                    "type": "openai_compatible",
                    "base_url": "http://first:8000/v1",
                },
            },
        }
        client = factory.from_config(config)
        assert isinstance(client.backend, OpenAICompatibleBackend)
        client.close()

    def test_from_config_raises_on_missing_default(self, mock_lg: Logger) -> None:
        """Test from_config raises when default backend not found."""
        factory = Factory(mock_lg)
        config = {
            "default": "missing",
            "backends": {
                "local": {"type": "openai_compatible"},
            },
        }
        with pytest.raises(ValueError, match="missing.*not found"):
            factory.from_config(config)


class TestLLMClientSyncAPI:
    """Test LLMClient sync API."""

    def test_chat_returns_content(self) -> None:
        """Test chat() returns content string."""
        response = ChatResponse(content="Hello!")
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result == "Hello!"

    def test_chat_full_returns_response(self) -> None:
        """Test chat_full() returns ChatResponse."""
        usage = ChatCompletionUsage(
            prompt_tokens=5, completion_tokens=2, total_tokens=7
        )
        response = ChatResponse(
            content="Hello!", usage=usage, finish_reason=FinishReason.STOP
        )
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend)

        result = client.chat_full(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Hello!"
        assert result.usage is not None
        assert result.usage.total_tokens == 7

    def test_chat_stream_yields_tokens(self) -> None:
        """Test chat_stream() yields tokens."""
        response = ChatResponse(content="Hello")
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend)

        tokens = list(client.chat_stream(messages=[{"role": "user", "content": "Hi"}]))

        assert tokens == ["H", "e", "l", "l", "o"]

    def test_last_response_available_after_chat(self) -> None:
        """Test last_response is available after chat."""
        response = ChatResponse(content="Hello!")
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend)

        client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert client.last_response is not None
        assert client.last_response.content == "Hello!"


class TestLLMClientAsyncAPI:
    """Test LLMClient async API."""

    @pytest.mark.asyncio
    async def test_chat_async_returns_content(self) -> None:
        """Test chat_async() returns content string."""
        response = ChatResponse(content="Hello!")
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend)

        result = await client.chat_async(messages=[{"role": "user", "content": "Hi"}])

        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_full_async_returns_response(self) -> None:
        """Test chat_full_async() returns ChatResponse."""
        response = ChatResponse(content="Hello!", finish_reason=FinishReason.STOP)
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend)

        result = await client.chat_full_async(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert result.content == "Hello!"
        assert result.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_chat_stream_async_yields_tokens(self) -> None:
        """Test chat_stream_async() yields tokens."""
        response = ChatResponse(content="Hello")
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend)

        tokens = []
        async for token in client.chat_stream_async(
            messages=[{"role": "user", "content": "Hi"}]
        ):
            tokens.append(token)

        assert tokens == ["H", "e", "l", "l", "o"]


class TestLLMClientResourceManagement:
    """Test resource management."""

    def test_sync_context_manager(self) -> None:
        """Test sync context manager calls close."""
        backend = MockBackend()
        with LLMClient(backend=backend) as client:
            assert client.backend is backend

        assert backend._closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async context manager calls aclose."""
        backend = MockBackend()
        async with LLMClient(backend=backend) as client:
            assert client.backend is backend

        assert backend._aclosed

    def test_close_delegates_to_backend(self) -> None:
        """Test close() calls backend.close()."""
        backend = MockBackend()
        client = LLMClient(backend=backend)

        client.close()

        assert backend._closed

    @pytest.mark.asyncio
    async def test_aclose_delegates_to_backend(self) -> None:
        """Test aclose() calls backend.aclose()."""
        backend = MockBackend()
        client = LLMClient(backend=backend)

        await client.aclose()

        assert backend._aclosed


class TestLLMClientDefaultModel:
    """Test default model handling."""

    def test_uses_default_model_when_not_specified(self) -> None:
        """Test default model is used when model not specified in call."""
        response = ChatResponse(content="Hello!")
        backend = MagicMock(spec=Backend)
        backend.chat.return_value = response
        backend.last_response = response

        client = LLMClient(backend=backend, default_model="gpt-4")
        client.chat(messages=[{"role": "user", "content": "Hi"}])

        # Verify model was passed to backend
        call_kwargs = backend.chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"

    def test_explicit_model_overrides_default(self) -> None:
        """Test explicit model overrides default."""
        response = ChatResponse(content="Hello!")
        backend = MagicMock(spec=Backend)
        backend.chat.return_value = response
        backend.last_response = response

        client = LLMClient(backend=backend, default_model="gpt-4")
        client.chat(messages=[{"role": "user", "content": "Hi"}], model="gpt-3.5")

        call_kwargs = backend.chat.call_args.kwargs
        assert call_kwargs["model"] == "gpt-3.5"
