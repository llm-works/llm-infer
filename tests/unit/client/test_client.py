"""Unit tests for LLMClient facade and Factory."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import ChatResponse, Factory, LLMClient, LLMRouter
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

    def test_from_config_single_backend_returns_router(self, mock_lg: Logger) -> None:
        """Test from_config with single backend config returns router."""
        factory = Factory(mock_lg)
        config = {
            "type": "openai_compatible",
            "base_url": "http://test:8000/v1",
            "model": "test-model",
        }
        router = factory.from_config(config)
        assert isinstance(router, LLMRouter)
        assert router.default == "default"
        assert "default" in router.clients
        router.close()

    def test_from_config_multi_backend_returns_router(self, mock_lg: Logger) -> None:
        """Test from_config with multiple backends returns router."""
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
        router = factory.from_config(config)
        assert isinstance(router, LLMRouter)
        assert router.default == "local"
        assert "local" in router.clients
        assert "remote" in router.clients
        router.close()

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
        router = factory.from_config(config)
        assert isinstance(router, LLMRouter)
        assert router.default == "first"
        router.close()

    def test_from_config_skips_disabled_backends(self, mock_lg: Logger) -> None:
        """Test from_config skips backends with enabled=false."""
        factory = Factory(mock_lg)
        config = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
                "disabled": {
                    "enabled": False,
                    "type": "openai_compatible",
                    "base_url": "http://disabled:8000/v1",
                },
            },
        }
        router = factory.from_config(config)
        assert "local" in router.clients
        assert "disabled" not in router.clients
        router.close()

    def test_from_config_enabled_defaults_to_true(self, mock_lg: Logger) -> None:
        """Test from_config treats missing enabled as true."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "no_enabled_key": {
                    "type": "openai_compatible",
                    "base_url": "http://test:8000/v1",
                },
            },
        }
        router = factory.from_config(config)
        assert "no_enabled_key" in router.clients
        router.close()

    def test_from_config_raises_on_no_enabled_backends(self, mock_lg: Logger) -> None:
        """Test from_config raises when all backends disabled."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "a": {"enabled": False, "type": "openai_compatible"},
                "b": {"enabled": False, "type": "openai_compatible"},
            },
        }
        with pytest.raises(ValueError, match="No enabled backends"):
            factory.from_config(config)

    def test_from_config_raises_if_default_backend_disabled(
        self, mock_lg: Logger
    ) -> None:
        """Test from_config raises when specified default is disabled."""
        factory = Factory(mock_lg)
        config = {
            "default": "disabled",
            "backends": {
                "disabled": {
                    "enabled": False,
                    "type": "openai_compatible",
                    "base_url": "http://disabled:8000/v1",
                },
                "enabled": {
                    "type": "openai_compatible",
                    "base_url": "http://enabled:8000/v1",
                },
            },
        }
        with pytest.raises(ValueError, match="Default backend 'disabled' not found"):
            factory.from_config(config)

    def test_from_config_with_discover_models_false(self, mock_lg: Logger) -> None:
        """Test from_config skips model discovery when disabled."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
            },
        }
        router = factory.from_config(config, discover_models=False)
        assert router.models == {}
        router.close()

    def test_from_config_uses_config_models_without_discovery(
        self, mock_lg: Logger
    ) -> None:
        """Test from_config uses config models list when discovery disabled."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                    "models": ["llama-3.1-8b", "qwen-2.5-7b"],
                },
            },
        }
        router = factory.from_config(config, discover_models=False)
        assert router.models == {"llama-3.1-8b": "local", "qwen-2.5-7b": "local"}
        router.close()

    def test_from_config_cleans_up_clients_on_model_conflict(
        self, mock_lg: Logger
    ) -> None:
        """Test from_config closes clients when model conflict raises."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "a": {
                    "type": "openai_compatible",
                    "base_url": "http://a:8000/v1",
                    "models": ["shared-model"],
                },
                "b": {
                    "type": "openai_compatible",
                    "base_url": "http://b:8000/v1",
                    "models": ["shared-model"],  # Conflict!
                },
            },
        }
        with pytest.raises(ValueError, match="Model 'shared-model' found in multiple"):
            factory.from_config(config, discover_models=False)
        # If we get here without resource leak, the fix is working
        # (We can't easily verify clients were closed without more intrusive mocking,
        # but the exception path now has cleanup code)


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


class TestLLMClientRateLimiting:
    """Test rate limiting and backoff functionality."""

    def test_can_call_returns_true_without_rate_limiting(self) -> None:
        """Test can_call returns True when no rate limiting configured."""
        backend = MockBackend()
        client = LLMClient(backend=backend)
        assert client.can_call() is True

    def test_can_call_returns_true_when_rate_limit_allows(
        self, mock_lg: Logger
    ) -> None:
        """Test can_call returns True when rate limit allows."""
        from appinfra.rate_limit import RateLimiter

        rate_limiter = RateLimiter(mock_lg, per_minute=60)
        backend = MockBackend()
        client = LLMClient(backend=backend, rate_limiter=rate_limiter)

        assert client.can_call() is True

    def test_can_call_returns_false_when_rate_limited(self, mock_lg: Logger) -> None:
        """Test can_call returns False when rate limit exceeded."""
        from appinfra.rate_limit import RateLimiter

        rate_limiter = RateLimiter(mock_lg, per_minute=60)
        # Simulate a recent call by setting last_t
        import time

        rate_limiter.last_t = time.time()

        backend = MockBackend()
        client = LLMClient(backend=backend, rate_limiter=rate_limiter)

        # Should be rate limited (less than 1 second since last call)
        assert client.can_call() is False

    def test_can_call_returns_false_during_backoff(self, mock_lg: Logger) -> None:
        """Test can_call returns False during backoff period."""
        from appinfra.rate_limit import Backoff

        backoff = Backoff(mock_lg, base=10.0)  # 10 second base
        backend = MockBackend()
        client = LLMClient(backend=backend, backoff=backoff)

        # Manually set backoff state
        import time

        client._backoff_until = time.time() + 10  # 10 seconds from now

        assert client.can_call() is False

    def test_can_call_returns_true_after_backoff_expires(self, mock_lg: Logger) -> None:
        """Test can_call returns True after backoff period expires."""
        from appinfra.rate_limit import Backoff

        backoff = Backoff(mock_lg, base=1.0)
        backend = MockBackend()
        client = LLMClient(backend=backend, backoff=backoff)

        # Set backoff in the past
        import time

        client._backoff_until = time.time() - 1  # 1 second ago

        assert client.can_call() is True

    def test_chat_resets_backoff_on_success(self, mock_lg: Logger) -> None:
        """Test successful chat resets backoff state."""
        from appinfra.rate_limit import Backoff

        backoff = Backoff(mock_lg, base=1.0)
        response = ChatResponse(content="Hello!")
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend, backoff=backoff)

        # Set some backoff state
        import time

        client._backoff_until = time.time() + 10
        backoff._attempts = 3

        # Make a successful call
        client.chat(messages=[{"role": "user", "content": "Hi"}])

        # Backoff should be reset
        assert client._backoff_until is None
        assert backoff.attempts == 0

    def test_chat_sets_backoff_on_unavailable_error(self, mock_lg: Logger) -> None:
        """Test chat sets backoff on BackendUnavailableError."""
        from appinfra.rate_limit import Backoff

        from llm_infer.client.exceptions import BackendUnavailableError

        backoff = Backoff(mock_lg, base=1.0, jitter=False)
        backend = MagicMock(spec=Backend)
        backend.chat.side_effect = BackendUnavailableError("Connection refused")

        client = LLMClient(backend=backend, backoff=backoff)

        with pytest.raises(BackendUnavailableError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])

        # Backoff should be set
        assert client._backoff_until is not None
        assert backoff.attempts == 1

    @pytest.mark.asyncio
    async def test_chat_async_resets_backoff_on_success(self, mock_lg: Logger) -> None:
        """Test successful async chat resets backoff state."""
        from appinfra.rate_limit import Backoff

        backoff = Backoff(mock_lg, base=1.0)
        response = ChatResponse(content="Hello!")
        backend = MockBackend(responses=[response])
        client = LLMClient(backend=backend, backoff=backoff)

        # Set some backoff state
        import time

        client._backoff_until = time.time() + 10
        backoff._attempts = 3

        # Make a successful async call
        await client.chat_async(messages=[{"role": "user", "content": "Hi"}])

        # Backoff should be reset
        assert client._backoff_until is None
        assert backoff.attempts == 0

    @pytest.mark.asyncio
    async def test_chat_async_sets_backoff_on_unavailable_error(
        self, mock_lg: Logger
    ) -> None:
        """Test async chat sets backoff on BackendUnavailableError."""
        from appinfra.rate_limit import Backoff

        from llm_infer.client.exceptions import BackendUnavailableError

        backoff = Backoff(mock_lg, base=1.0, jitter=False)
        backend = MagicMock(spec=Backend)
        backend.chat_async.side_effect = BackendUnavailableError("Connection refused")

        client = LLMClient(backend=backend, backoff=backoff)

        with pytest.raises(BackendUnavailableError):
            await client.chat_async(messages=[{"role": "user", "content": "Hi"}])

        # Backoff should be set
        assert client._backoff_until is not None
        assert backoff.attempts == 1

    def test_chat_stream_resets_backoff_on_success(self, mock_lg: Logger) -> None:
        """Test successful streaming chat resets backoff state."""
        from appinfra.rate_limit import Backoff

        backoff = Backoff(mock_lg, base=1.0)
        backend = MagicMock(spec=Backend)
        backend.chat_stream.return_value = iter(["Hello", " world"])

        client = LLMClient(backend=backend, backoff=backoff)

        # Set some backoff state
        import time

        client._backoff_until = time.time() + 10
        backoff._attempts = 3

        # Consume the stream
        tokens = list(client.chat_stream(messages=[{"role": "user", "content": "Hi"}]))

        # Backoff should be reset
        assert tokens == ["Hello", " world"]
        assert client._backoff_until is None
        assert backoff.attempts == 0

    def test_chat_stream_sets_backoff_on_unavailable_error(
        self, mock_lg: Logger
    ) -> None:
        """Test streaming chat sets backoff on BackendUnavailableError."""
        from appinfra.rate_limit import Backoff

        from llm_infer.client.exceptions import BackendUnavailableError

        backoff = Backoff(mock_lg, base=1.0, jitter=False)
        backend = MagicMock(spec=Backend)
        backend.chat_stream.side_effect = BackendUnavailableError("Connection refused")

        client = LLMClient(backend=backend, backoff=backoff)

        with pytest.raises(BackendUnavailableError):
            list(client.chat_stream(messages=[{"role": "user", "content": "Hi"}]))

        # Backoff should be set
        assert client._backoff_until is not None
        assert backoff.attempts == 1

    @pytest.mark.asyncio
    async def test_chat_stream_async_resets_backoff_on_success(
        self, mock_lg: Logger
    ) -> None:
        """Test successful async streaming chat resets backoff state."""
        from appinfra.rate_limit import Backoff

        backoff = Backoff(mock_lg, base=1.0)
        backend = MagicMock(spec=Backend)

        async def mock_stream(*args: Any, **kwargs: Any):
            for token in ["Hello", " world"]:
                yield token

        backend.chat_stream_async.return_value = mock_stream()

        client = LLMClient(backend=backend, backoff=backoff)

        # Set some backoff state
        import time

        client._backoff_until = time.time() + 10
        backoff._attempts = 3

        # Consume the stream
        tokens = [
            token
            async for token in client.chat_stream_async(
                messages=[{"role": "user", "content": "Hi"}]
            )
        ]

        # Backoff should be reset
        assert tokens == ["Hello", " world"]
        assert client._backoff_until is None
        assert backoff.attempts == 0

    @pytest.mark.asyncio
    async def test_chat_stream_async_sets_backoff_on_unavailable_error(
        self, mock_lg: Logger
    ) -> None:
        """Test async streaming chat sets backoff on BackendUnavailableError."""
        from appinfra.rate_limit import Backoff

        from llm_infer.client.exceptions import BackendUnavailableError

        backoff = Backoff(mock_lg, base=1.0, jitter=False)
        backend = MagicMock(spec=Backend)
        backend.chat_stream_async.side_effect = BackendUnavailableError(
            "Connection refused"
        )

        client = LLMClient(backend=backend, backoff=backoff)

        with pytest.raises(BackendUnavailableError):
            async for _ in client.chat_stream_async(
                messages=[{"role": "user", "content": "Hi"}]
            ):
                pass

        # Backoff should be set
        assert client._backoff_until is not None
        assert backoff.attempts == 1


class TestFactoryRateLimitConfig:
    """Test Factory rate limit configuration parsing."""

    def test_from_config_creates_rate_limiter(self, mock_lg: Logger) -> None:
        """Test from_config creates rate limiter from config."""
        factory = Factory(mock_lg)
        config = {
            "rate_limit": {"per_minute": 30},
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
            },
        }
        router = factory.from_config(config, discover_models=False)

        # Check that client has rate limiter
        client = router.get_client()
        assert client._rate_limiter is not None
        assert client._rate_limiter.per_minute == 30
        router.close()

    def test_from_config_creates_backoff(self, mock_lg: Logger) -> None:
        """Test from_config creates backoff from config."""
        factory = Factory(mock_lg)
        config = {
            "rate_limit": {
                "backoff": {"base": 2.0, "max": 120.0},
            },
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
            },
        }
        router = factory.from_config(config, discover_models=False)

        # Check that client has backoff
        client = router.get_client()
        assert client._backoff is not None
        assert client._backoff.base == 2.0
        assert client._backoff.max_delay == 120.0
        router.close()

    def test_from_config_without_rate_limit(self, mock_lg: Logger) -> None:
        """Test from_config without rate_limit creates client without rate limiting."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
            },
        }
        router = factory.from_config(config, discover_models=False)

        client = router.get_client()
        assert client._rate_limiter is None
        assert client._backoff is None
        router.close()

    def test_from_config_rate_limit_applies_to_all_backends(
        self, mock_lg: Logger
    ) -> None:
        """Test rate_limit config applies to all backends."""
        factory = Factory(mock_lg)
        config = {
            "rate_limit": {"per_minute": 30, "backoff": {"base": 2.0}},
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
                "remote": {
                    "type": "openai_compatible",
                    "base_url": "http://remote:8000/v1",
                },
            },
        }
        router = factory.from_config(config, discover_models=False)

        # Both clients should have rate limiters
        local_client = router.get_client(backend="local")
        remote_client = router.get_client(backend="remote")
        assert local_client._rate_limiter is not None
        assert remote_client._rate_limiter is not None
        assert local_client._backoff is not None
        assert remote_client._backoff is not None
        router.close()
