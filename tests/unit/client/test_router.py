"""Unit tests for LLMRouter."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import ChatResponse, LLMClient, LLMRouter
from llm_infer.client.backends import Backend

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


def make_client(responses: list[ChatResponse] | None = None) -> LLMClient:
    """Create a client with a mock backend."""
    backend = MockBackend(responses=responses)
    return LLMClient(backend=backend)


class TestLLMRouterInit:
    """Test LLMRouter initialization."""

    def test_init_with_clients(self, mock_lg: Logger) -> None:
        """Test router initializes with clients dict."""
        client_a = make_client()
        client_b = make_client()
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        assert router.clients == {"a": client_a, "b": client_b}
        assert router.default == "a"

    def test_init_raises_if_default_not_in_clients(self, mock_lg: Logger) -> None:
        """Test router raises if default not in clients."""
        client_a = make_client()
        with pytest.raises(
            ValueError, match="Default backend 'missing' not in clients"
        ):
            LLMRouter(mock_lg, {"a": client_a}, "missing")

    def test_init_raises_if_model_routes_to_unknown_backend(
        self, mock_lg: Logger
    ) -> None:
        """Test router raises if model routing references unknown backend."""
        client_a = make_client()
        model_to_backend = {"model-x": "unknown"}
        with pytest.raises(
            ValueError, match="Model 'model-x' routes to unknown backend 'unknown'"
        ):
            LLMRouter(mock_lg, {"a": client_a}, "a", model_to_backend)


class TestLLMRouterRouting:
    """Test LLMRouter routing behavior."""

    def test_chat_uses_default_backend(self, mock_lg: Logger) -> None:
        """Test chat() uses default backend when no backend specified."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client([response_a])
        client_b = make_client([response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.chat([{"role": "user", "content": "Hi"}])

        assert result == "From A"

    def test_chat_routes_to_specified_backend(self, mock_lg: Logger) -> None:
        """Test chat() routes to specified backend."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client([response_a])
        client_b = make_client([response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.chat([{"role": "user", "content": "Hi"}], backend="b")

        assert result == "From B"

    def test_chat_raises_on_unknown_backend(self, mock_lg: Logger) -> None:
        """Test chat() raises on unknown backend."""
        client_a = make_client([ChatResponse(content="A")])
        router = LLMRouter(mock_lg, {"a": client_a}, "a")

        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            router.chat([{"role": "user", "content": "Hi"}], backend="unknown")

    def test_chat_full_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_full() routes to correct backend."""
        response = ChatResponse(content="Full response")
        client = make_client([response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.chat_full([{"role": "user", "content": "Hi"}])

        assert result.content == "Full response"

    def test_chat_stream_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_stream() routes to correct backend."""
        response = ChatResponse(content="ABC")
        client = make_client([response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        tokens = list(router.chat_stream([{"role": "user", "content": "Hi"}]))

        assert tokens == ["A", "B", "C"]


class TestLLMRouterAsync:
    """Test LLMRouter async API."""

    @pytest.mark.asyncio
    async def test_chat_async_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_async() routes to correct backend."""
        response = ChatResponse(content="Async response")
        client = make_client([response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = await router.chat_async([{"role": "user", "content": "Hi"}])

        assert result == "Async response"

    @pytest.mark.asyncio
    async def test_chat_async_routes_to_specified_backend(
        self, mock_lg: Logger
    ) -> None:
        """Test chat_async() routes to specified backend."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client([response_a])
        client_b = make_client([response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = await router.chat_async(
            [{"role": "user", "content": "Hi"}], backend="b"
        )

        assert result == "From B"

    @pytest.mark.asyncio
    async def test_chat_full_async_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_full_async() routes correctly."""
        response = ChatResponse(content="Full async")
        client = make_client([response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = await router.chat_full_async([{"role": "user", "content": "Hi"}])

        assert result.content == "Full async"

    @pytest.mark.asyncio
    async def test_chat_stream_async_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_stream_async() routes correctly."""
        response = ChatResponse(content="XYZ")
        client = make_client([response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        tokens = []
        async for token in router.chat_stream_async(
            [{"role": "user", "content": "Hi"}]
        ):
            tokens.append(token)

        assert tokens == ["X", "Y", "Z"]


class TestLLMRouterModelRouting:
    """Test LLMRouter model-based routing."""

    def test_routes_by_model_when_in_routing_table(self, mock_lg: Logger) -> None:
        """Test routing by model when model is in the routing table."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client([response_a])
        client_b = make_client([response_b])
        model_to_backend = {"model-a": "a", "model-b": "b"}
        router = LLMRouter(
            mock_lg, {"a": client_a, "b": client_b}, "a", model_to_backend
        )

        result = router.chat([{"role": "user", "content": "Hi"}], model="model-b")

        assert result == "From B"

    def test_falls_back_to_default_when_model_not_in_table(
        self, mock_lg: Logger
    ) -> None:
        """Test fallback to default when model not in routing table."""
        response_a = ChatResponse(content="From A")
        client_a = make_client([response_a])
        model_to_backend = {"known-model": "a"}
        router = LLMRouter(mock_lg, {"a": client_a}, "a", model_to_backend)

        result = router.chat([{"role": "user", "content": "Hi"}], model="unknown-model")

        assert result == "From A"  # Falls back to default

    def test_explicit_backend_takes_priority_over_model(self, mock_lg: Logger) -> None:
        """Test explicit backend param overrides model-based routing."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client([response_a])
        client_b = make_client([response_b])
        model_to_backend = {"model-b": "b"}
        router = LLMRouter(
            mock_lg, {"a": client_a, "b": client_b}, "a", model_to_backend
        )

        # model="model-b" would route to b, but backend="a" overrides
        result = router.chat(
            [{"role": "user", "content": "Hi"}], model="model-b", backend="a"
        )

        assert result == "From A"

    def test_models_property_returns_routing_table(self, mock_lg: Logger) -> None:
        """Test models property exposes the routing table."""
        client = make_client()
        model_to_backend = {"model-x": "main", "model-y": "main"}
        router = LLMRouter(mock_lg, {"main": client}, "main", model_to_backend)

        assert router.models == {"model-x": "main", "model-y": "main"}

    def test_get_client_with_model_param(self, mock_lg: Logger) -> None:
        """Test get_client resolves by model."""
        client_a = make_client()
        client_b = make_client()
        model_to_backend = {"gpt-4": "b"}
        router = LLMRouter(
            mock_lg, {"a": client_a, "b": client_b}, "a", model_to_backend
        )

        resolved = router.get_client(model="gpt-4")

        assert resolved is client_b


class TestLLMRouterResourceManagement:
    """Test LLMRouter resource management."""

    def test_close_closes_all_clients(self, mock_lg: Logger) -> None:
        """Test close() closes all clients."""
        backend_a = MockBackend()
        backend_b = MockBackend()
        client_a = LLMClient(backend=backend_a)
        client_b = LLMClient(backend=backend_b)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        router.close()

        assert backend_a._closed
        assert backend_b._closed

    @pytest.mark.asyncio
    async def test_aclose_closes_all_clients(self, mock_lg: Logger) -> None:
        """Test aclose() closes all clients."""
        backend_a = MockBackend()
        backend_b = MockBackend()
        client_a = LLMClient(backend=backend_a)
        client_b = LLMClient(backend=backend_b)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        await router.aclose()

        assert backend_a._aclosed
        assert backend_b._aclosed

    def test_sync_context_manager(self, mock_lg: Logger) -> None:
        """Test sync context manager calls close."""
        backend = MockBackend()
        client = LLMClient(backend=backend)

        with LLMRouter(mock_lg, {"main": client}, "main") as router:
            assert router.clients == {"main": client}

        assert backend._closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_lg: Logger) -> None:
        """Test async context manager calls aclose."""
        backend = MockBackend()
        client = LLMClient(backend=backend)

        async with LLMRouter(mock_lg, {"main": client}, "main") as router:
            assert router.clients == {"main": client}

        assert backend._aclosed


class TestLLMRouterCanCall:
    """Test LLMRouter.can_call() method."""

    def test_can_call_delegates_to_default_client(self, mock_lg: Logger) -> None:
        """Test can_call() delegates to default client."""
        backend = MockBackend()
        client = LLMClient(backend=backend)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        # Client has no rate limiting, should return True
        assert router.can_call() is True

    def test_can_call_delegates_to_specified_backend(self, mock_lg: Logger) -> None:
        """Test can_call(backend=...) delegates to specified client."""
        from appinfra.rate_limit import RateLimiter

        backend_a = MockBackend()
        backend_b = MockBackend()

        # Client A has rate limiting
        rate_limiter = RateLimiter(mock_lg, per_minute=60)
        import time

        rate_limiter.last_t = time.time()  # Simulate recent call

        client_a = LLMClient(backend=backend_a, rate_limiter=rate_limiter)
        client_b = LLMClient(backend=backend_b)  # No rate limiting

        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        # Default (a) should be rate limited
        assert router.can_call() is False
        # Backend b should be allowed
        assert router.can_call(backend="b") is True

    def test_can_call_with_model_routing(self, mock_lg: Logger) -> None:
        """Test can_call(model=...) uses model routing."""
        from appinfra.rate_limit import Backoff

        backend_a = MockBackend()
        backend_b = MockBackend()

        # Client B has backoff active
        backoff = Backoff(mock_lg, base=10.0)
        client_a = LLMClient(backend=backend_a)
        client_b = LLMClient(backend=backend_b, backoff=backoff)

        import time

        client_b._backoff_until = time.time() + 10  # Active backoff

        model_routing = {"model-a": "a", "model-b": "b"}
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a", model_routing)

        # Model-a routes to client_a (no backoff)
        assert router.can_call(model="model-a") is True
        # Model-b routes to client_b (has backoff)
        assert router.can_call(model="model-b") is False

    def test_can_call_raises_on_unknown_backend(self, mock_lg: Logger) -> None:
        """Test can_call raises ValueError for unknown backend."""
        backend = MockBackend()
        client = LLMClient(backend=backend)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            router.can_call(backend="unknown")
