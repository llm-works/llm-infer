"""Unit tests for LLMClient facade and Factory."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import (
    ChatRequest,
    ChatResponse,
    Factory,
    LLMClient,
    LLMRouter,
    ModelConflictError,
)
from llm_infer.client.backends import Backend, BackendContext, OpenAICompatibleBackend
from llm_infer.schemas.openai import ChatCompletionUsage, FinishReason

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger for testing."""
    return MagicMock(spec=Logger)


class MockBackend(Backend):
    """Mock backend for testing."""

    def __init__(
        self,
        lg: Logger,
        name: str,
        ctx: BackendContext | None = None,
        default_model: str | None = None,
        responses: list[ChatResponse] | None = None,
    ) -> None:
        super().__init__(lg, name, ctx, default_model)
        self._responses = iter(responses or [])
        self._last_response: ChatResponse | None = None
        self._closed = False
        self._aclosed = False

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    @property
    def provider(self) -> str:
        return "mock"

    def chat(self, request: ChatRequest) -> ChatResponse:
        if self._ctx.rate_limiter is not None:
            self._ctx.rate_limiter.next()
        response = next(self._responses)
        self._last_response = response
        return response

    def chat_stream(self, request: ChatRequest) -> Iterator[str]:
        if self._ctx.rate_limiter is not None:
            self._ctx.rate_limiter.next()
        response = next(self._responses)
        yield from response.content
        self._last_response = response

    async def chat_async(self, request: ChatRequest) -> ChatResponse:
        if self._ctx.rate_limiter is not None:
            await asyncio.to_thread(self._ctx.rate_limiter.next)
        response = next(self._responses)
        self._last_response = response
        return response

    async def chat_stream_async(self, request: ChatRequest) -> AsyncIterator[str]:
        if self._ctx.rate_limiter is not None:
            await asyncio.to_thread(self._ctx.rate_limiter.next)
        response = next(self._responses)
        for char in response.content:
            yield char
        self._last_response = response

    def close(self) -> None:
        self._closed = True

    async def aclose(self) -> None:
        self._aclosed = True


class TestLLMClientInit:
    """Test LLMClient initialization."""

    def test_init_with_backend(self, mock_lg: Logger) -> None:
        """Test client initializes with backend."""
        backend = MockBackend(mock_lg, "test")
        client = LLMClient(lg=mock_lg, backend=backend)
        assert client.backend is backend
        assert client.last_response is None

    def test_init_with_default_model(self, mock_lg: Logger) -> None:
        """Test client stores default model via backend."""
        backend = MockBackend(mock_lg, "test", default_model="gpt-4")
        client = LLMClient(lg=mock_lg, backend=backend)
        assert client.default_model == "gpt-4"


class TestFactory:
    """Test Factory methods."""

    def test_openai_creates_openai_backend(self, mock_lg: Logger) -> None:
        """Test openai() creates OpenAI backend."""
        factory = Factory(mock_lg)
        client = factory.openai(
            base_url="http://test:8000/v1",
            default_model="test-model",
            api_key="test-key",
        )
        assert isinstance(client.backend, OpenAICompatibleBackend)
        assert client.default_model == "test-model"
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

    def test_from_config_model_key(self, mock_lg: Logger) -> None:
        """Test from_config reads 'model' config key."""
        factory = Factory(mock_lg)
        config = {
            "type": "openai_compatible",
            "base_url": "http://test:8000/v1",
            "model": "test-model",
        }
        router = factory.from_config(config)
        assert router.clients["default"].default_model == "test-model"
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
                    "models": ["shared-model"],
                },
            },
        }
        with pytest.raises(ModelConflictError) as exc_info:
            factory.from_config(config, discover_models=False)

        assert exc_info.value.model == "shared-model"


class TestLLMClientSyncAPI:
    """Test LLMClient sync API."""

    def test_chat_returns_response(self, mock_lg: Logger) -> None:
        """Test chat() returns ChatResponse."""
        usage = ChatCompletionUsage(
            prompt_tokens=5, completion_tokens=2, total_tokens=7
        )
        response = ChatResponse(
            content="Hello!", usage=usage, finish_reason=FinishReason.STOP
        )
        backend = MockBackend(mock_lg, "test", responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Hello!"
        assert result.usage is not None
        assert result.usage.total_tokens == 7

    def test_chat_stream_yields_tokens(self, mock_lg: Logger) -> None:
        """Test chat_stream() yields tokens."""
        response = ChatResponse(content="Hello")
        backend = MockBackend(mock_lg, "test", responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        tokens = list(client.chat_stream(messages=[{"role": "user", "content": "Hi"}]))

        assert tokens == ["H", "e", "l", "l", "o"]

    def test_last_response_available_after_chat(self, mock_lg: Logger) -> None:
        """Test last_response is available after chat."""
        response = ChatResponse(content="Hello!")
        backend = MockBackend(mock_lg, "test", responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert client.last_response is not None
        assert client.last_response.content == "Hello!"


class TestLLMClientAsyncAPI:
    """Test LLMClient async API."""

    @pytest.mark.asyncio
    async def test_chat_async_returns_response(self, mock_lg: Logger) -> None:
        """Test chat_async() returns ChatResponse."""
        response = ChatResponse(content="Hello!")
        backend = MockBackend(mock_lg, "test", responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = await client.chat_async(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_stream_async_yields_tokens(self, mock_lg: Logger) -> None:
        """Test chat_stream_async() yields tokens."""
        response = ChatResponse(content="Hello")
        backend = MockBackend(mock_lg, "test", responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        tokens = []
        async for token in client.chat_stream_async(
            messages=[{"role": "user", "content": "Hi"}]
        ):
            tokens.append(token)

        assert tokens == ["H", "e", "l", "l", "o"]


class TestLLMClientResourceManagement:
    """Test LLMClient resource management."""

    def test_sync_context_manager(self, mock_lg: Logger) -> None:
        """Test sync context manager closes client."""
        backend = MockBackend(mock_lg, "test", responses=[ChatResponse(content="")])
        with LLMClient(lg=mock_lg, backend=backend) as client:
            assert client.backend is backend
        assert backend._closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_lg: Logger) -> None:
        """Test async context manager closes client."""
        backend = MockBackend(mock_lg, "test", responses=[ChatResponse(content="")])
        async with LLMClient(lg=mock_lg, backend=backend) as client:
            assert client.backend is backend
        assert backend._aclosed

    def test_close_delegates_to_backend(self, mock_lg: Logger) -> None:
        """Test close() closes backend."""
        backend = MockBackend(mock_lg, "test")
        client = LLMClient(lg=mock_lg, backend=backend)
        client.close()
        assert backend._closed

    @pytest.mark.asyncio
    async def test_aclose_delegates_to_backend(self, mock_lg: Logger) -> None:
        """Test aclose() closes backend."""
        backend = MockBackend(mock_lg, "test")
        client = LLMClient(lg=mock_lg, backend=backend)
        await client.aclose()
        assert backend._aclosed


class TestLLMClientDefaultModel:
    """Test default model handling."""

    def test_uses_default_model_when_not_specified(self, mock_lg: Logger) -> None:
        """Test default model is used when model not specified."""
        response = ChatResponse(content="Hello!")
        backend = MockBackend(
            mock_lg, "test", default_model="default-model", responses=[response]
        )
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Hello!"
        assert client.default_model == "default-model"

    def test_explicit_model_overrides_default(self, mock_lg: Logger) -> None:
        """Test explicit model overrides default."""
        response = ChatResponse(content="Hello!")
        backend = MockBackend(
            mock_lg, "test", default_model="default-model", responses=[response]
        )
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="explicit-model",
        )

        assert result.content == "Hello!"


class TestLLMClientRateLimiting:
    """Test rate limiting integration."""

    def test_can_call_returns_true_without_rate_limiting(self, mock_lg: Logger) -> None:
        """Test can_call() returns True without rate limiter."""
        backend = MockBackend(mock_lg, "test")
        client = LLMClient(lg=mock_lg, backend=backend)

        assert client.can_call() is True

    def test_can_call_returns_true_when_rate_limit_allows(
        self, mock_lg: Logger
    ) -> None:
        """Test can_call() returns True when rate limiter allows."""
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.can_proceed.return_value = True

        ctx = BackendContext(rate_limiter=mock_rate_limiter)
        backend = MockBackend(mock_lg, "test", ctx=ctx)
        client = LLMClient(lg=mock_lg, backend=backend)

        assert client.can_call() is True
        mock_rate_limiter.can_proceed.assert_called_once()

    def test_can_call_returns_false_when_rate_limited(self, mock_lg: Logger) -> None:
        """Test can_call() returns False when rate limited."""
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.can_proceed.return_value = False

        ctx = BackendContext(rate_limiter=mock_rate_limiter)
        backend = MockBackend(mock_lg, "test", ctx=ctx)
        client = LLMClient(lg=mock_lg, backend=backend)

        assert client.can_call() is False

    def test_rate_limiter_enforced_on_chat(self, mock_lg: Logger) -> None:
        """Test rate limiter is called during chat."""
        mock_rate_limiter = MagicMock()
        response = ChatResponse(content="Hello!")

        ctx = BackendContext(rate_limiter=mock_rate_limiter)
        backend = MockBackend(mock_lg, "test", ctx=ctx, responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        client.chat(messages=[{"role": "user", "content": "Hi"}])

        mock_rate_limiter.next.assert_called_once()


class TestLLMClientRetry:
    """Test retry behavior with backoff."""

    def test_retry_on_429_rate_limited(self, mock_lg: Logger) -> None:
        """Test retry on 429 rate limited error."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        response = ChatResponse(content="Success!")
        call_count = 0

        class RetryBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise BackendRequestError("Rate limited", status_code=429)
                return next(self._responses)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = RetryBackend(mock_lg, "test", ctx=ctx, responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Success!"
        assert call_count == 2

    def test_retry_on_500_internal_server_error(self, mock_lg: Logger) -> None:
        """Test retry on 500 internal server error."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        response = ChatResponse(content="Success!")
        call_count = 0

        class RetryBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise BackendRequestError("Internal error", status_code=500)
                return next(self._responses)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = RetryBackend(mock_lg, "test", ctx=ctx, responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Success!"
        assert call_count == 2

    def test_retry_on_503_service_unavailable(self, mock_lg: Logger) -> None:
        """Test retry on 503 service unavailable."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        response = ChatResponse(content="Success!")
        call_count = 0

        class RetryBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise BackendRequestError("Unavailable", status_code=503)
                return next(self._responses)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = RetryBackend(mock_lg, "test", ctx=ctx, responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Success!"
        assert call_count == 2

    def test_retry_on_529_overloaded(self, mock_lg: Logger) -> None:
        """Test retry on 529 overloaded."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        response = ChatResponse(content="Success!")
        call_count = 0

        class RetryBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise BackendRequestError("Overloaded", status_code=529)
                return next(self._responses)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = RetryBackend(mock_lg, "test", ctx=ctx, responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Success!"
        assert call_count == 2

    def test_retry_on_connection_failure(self, mock_lg: Logger) -> None:
        """Test retry on connection failure."""
        from llm_infer.client import BackendUnavailableError
        from llm_infer.client.backends import RetryConfig

        response = ChatResponse(content="Success!")
        call_count = 0

        class RetryBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise BackendUnavailableError("Connection refused")
                return next(self._responses)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = RetryBackend(mock_lg, "test", ctx=ctx, responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Success!"
        assert call_count == 2

    def test_retry_on_transport_error(self, mock_lg: Logger) -> None:
        """Test retry on transport error (no status code)."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        response = ChatResponse(content="Success!")
        call_count = 0

        class RetryBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise BackendRequestError("Transport error")
                return next(self._responses)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = RetryBackend(mock_lg, "test", ctx=ctx, responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Success!"
        assert call_count == 2

    def test_no_retry_on_non_transient_error(self, mock_lg: Logger) -> None:
        """Test no retry on non-transient error (4xx)."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        class RetryBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                raise BackendRequestError("Bad request", status_code=400)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = RetryBackend(mock_lg, "test", ctx=ctx)
        client = LLMClient(lg=mock_lg, backend=backend)

        with pytest.raises(BackendRequestError) as exc_info:
            client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert exc_info.value.status_code == 400

    def test_retry_timeout_exceeded(self, mock_lg: Logger) -> None:
        """Test retry stops after timeout."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        class RetryBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                raise BackendRequestError("Always fails", status_code=500)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1, timeout=0.01))
        backend = RetryBackend(mock_lg, "test", ctx=ctx)
        client = LLMClient(lg=mock_lg, backend=backend)

        with pytest.raises(BackendRequestError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])

    def test_no_retry_when_backoff_not_configured(self, mock_lg: Logger) -> None:
        """Test no retry when backoff not configured."""
        from llm_infer.client import BackendRequestError

        class FailingBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                raise BackendRequestError("Server error", status_code=500)

        backend = FailingBackend(mock_lg, "test")
        client = LLMClient(lg=mock_lg, backend=backend)

        with pytest.raises(BackendRequestError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_retry_async_on_transient_error(self, mock_lg: Logger) -> None:
        """Test async retry on transient error."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        response = ChatResponse(content="Success!")
        call_count = 0

        class RetryBackend(MockBackend):
            async def chat_async(self, request: ChatRequest) -> ChatResponse:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise BackendRequestError("Rate limited", status_code=429)
                return next(self._responses)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = RetryBackend(mock_lg, "test", ctx=ctx, responses=[response])
        client = LLMClient(lg=mock_lg, backend=backend)

        result = await client.chat_async(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Success!"
        assert call_count == 2

    def test_backoff_gatekeeper_on_non_transient_error(self, mock_lg: Logger) -> None:
        """Test backoff doesn't retry non-transient errors."""
        from llm_infer.client import BackendRequestError
        from llm_infer.client.backends import RetryConfig

        call_count = 0

        class FailingBackend(MockBackend):
            def chat(self, request: ChatRequest) -> ChatResponse:
                nonlocal call_count
                call_count += 1
                raise BackendRequestError("Auth failed", status_code=401)

        ctx = BackendContext(retry=RetryConfig(base=0.01, max_delay=0.1))
        backend = FailingBackend(mock_lg, "test", ctx=ctx)
        client = LLMClient(lg=mock_lg, backend=backend)

        with pytest.raises(BackendRequestError):
            client.chat(messages=[{"role": "user", "content": "Hi"}])

        assert call_count == 1


class TestFactoryRetryConfig:
    """Test Factory retry configuration."""

    def test_from_config_creates_retry(self, mock_lg: Logger) -> None:
        """Test from_config creates retry config."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
            },
            "retry": {
                "base": 2.0,
                "factor": 3.0,
                "max_delay": 120.0,
            },
        }
        router = factory.from_config(config, discover_models=False)
        client = router.get_client()
        assert client.backend.ctx.retry is not None
        assert client.backend.ctx.retry.base == 2.0
        assert client.backend.ctx.retry.factor == 3.0
        assert client.backend.ctx.retry.max_delay == 120.0
        router.close()

    def test_from_config_retry_disabled(self, mock_lg: Logger) -> None:
        """Test from_config with retry disabled."""
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
        assert client.backend.ctx.retry is None
        router.close()

    def test_from_config_retry_timeout(self, mock_lg: Logger) -> None:
        """Test from_config with retry timeout."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
            },
            "retry": {"timeout": 300.0},
        }
        router = factory.from_config(config, discover_models=False)
        client = router.get_client()
        assert client.backend.ctx.retry is not None
        assert client.backend.ctx.retry.timeout == 300.0
        router.close()

    def test_per_backend_retry_override(self, mock_lg: Logger) -> None:
        """Test per-backend retry config overrides global."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                    "retry": {"base": 5.0, "max_delay": 30.0},
                },
            },
            "retry": {"base": 1.0, "max_delay": 60.0},
        }
        router = factory.from_config(config, discover_models=False)
        client = router.get_client()
        assert client.backend.ctx.retry is not None
        assert client.backend.ctx.retry.base == 5.0
        assert client.backend.ctx.retry.max_delay == 30.0
        router.close()

    def test_per_backend_rate_limit_override(self, mock_lg: Logger) -> None:
        """Test per-backend rate limit config overrides global."""
        factory = Factory(mock_lg)
        config = {
            "backends": {
                "local": {
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                    "rate_limit": {"per_minute": 30},
                },
            },
            "rate_limit": {"per_minute": 60},
        }
        router = factory.from_config(config, discover_models=False)
        client = router.get_client()
        assert client.backend.ctx.rate_limiter is not None
        router.close()


class TestFromBackendConfig:
    """Test Factory.from_backend_config method."""

    def test_from_backend_config_with_rate_limit(self, mock_lg: Logger) -> None:
        """Test from_backend_config creates rate limiter."""
        factory = Factory(mock_lg)
        config = {
            "type": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "rate_limit": {"per_minute": 60},
        }
        client = factory.from_backend_config(config)
        assert client.backend.ctx.rate_limiter is not None
        client.close()

    def test_from_backend_config_with_retry(self, mock_lg: Logger) -> None:
        """Test from_backend_config creates retry config."""
        factory = Factory(mock_lg)
        config = {
            "type": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "retry": {"base": 2.0, "max_delay": 30.0},
        }
        client = factory.from_backend_config(config)
        assert client.backend.ctx.retry is not None
        assert client.backend.ctx.retry.base == 2.0
        client.close()

    def test_from_backend_config_with_retry_timeout(self, mock_lg: Logger) -> None:
        """Test from_backend_config with retry timeout."""
        factory = Factory(mock_lg)
        config = {
            "type": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "retry": {"timeout": 120.0},
        }
        client = factory.from_backend_config(config)
        assert client.backend.ctx.retry is not None
        assert client.backend.ctx.retry.timeout == 120.0
        client.close()

    def test_from_backend_config_with_both_rate_limit_and_retry(
        self, mock_lg: Logger
    ) -> None:
        """Test from_backend_config with both rate limit and retry."""
        factory = Factory(mock_lg)
        config = {
            "type": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "rate_limit": {"per_minute": 30},
            "retry": {"base": 1.0},
        }
        client = factory.from_backend_config(config)
        assert client.backend.ctx.rate_limiter is not None
        assert client.backend.ctx.retry is not None
        client.close()
