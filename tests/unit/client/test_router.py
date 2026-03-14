"""Unit tests for LLMRouter."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import ChatResponse, LLMClient, LLMRouter, ResolvedTarget
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


def make_client(lg: Logger, responses: list[ChatResponse] | None = None) -> LLMClient:
    """Create a client with a mock backend."""
    backend = MockBackend(responses=responses)
    return LLMClient(lg=lg, backend=backend)


class TestLLMRouterInit:
    """Test LLMRouter initialization."""

    def test_init_with_clients(self, mock_lg: Logger) -> None:
        """Test router initializes with clients dict."""
        client_a = make_client(mock_lg)
        client_b = make_client(mock_lg)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        assert router.clients == {"a": client_a, "b": client_b}
        assert router.default == "a"

    def test_init_raises_if_default_not_in_clients(self, mock_lg: Logger) -> None:
        """Test router raises if default not in clients."""
        client_a = make_client(mock_lg)
        with pytest.raises(
            ValueError, match="Default backend 'missing' not in clients"
        ):
            LLMRouter(mock_lg, {"a": client_a}, "missing")

    def test_init_raises_if_model_routes_to_unknown_backend(
        self, mock_lg: Logger
    ) -> None:
        """Test router raises if model routing references unknown backend."""
        client_a = make_client(mock_lg)
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
        client_a = make_client(mock_lg, [response_a])
        client_b = make_client(mock_lg, [response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.chat([{"role": "user", "content": "Hi"}])

        assert result.content == "From A"

    def test_chat_routes_to_specified_backend(self, mock_lg: Logger) -> None:
        """Test chat() routes to specified backend."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client(mock_lg, [response_a])
        client_b = make_client(mock_lg, [response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.chat([{"role": "user", "content": "Hi"}], backend="b")

        assert result.content == "From B"

    def test_chat_raises_on_unknown_backend(self, mock_lg: Logger) -> None:
        """Test chat() raises on unknown backend."""
        client_a = make_client(mock_lg, [ChatResponse(content="A")])
        router = LLMRouter(mock_lg, {"a": client_a}, "a")

        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            router.chat([{"role": "user", "content": "Hi"}], backend="unknown")

    def test_chat_stream_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_stream() routes to correct backend."""
        response = ChatResponse(content="ABC")
        client = make_client(mock_lg, [response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        tokens = list(router.chat_stream([{"role": "user", "content": "Hi"}]))

        assert tokens == ["A", "B", "C"]


class TestLLMRouterAsync:
    """Test LLMRouter async API."""

    @pytest.mark.asyncio
    async def test_chat_async_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_async() routes to correct backend."""
        response = ChatResponse(content="Async response")
        client = make_client(mock_lg, [response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = await router.chat_async([{"role": "user", "content": "Hi"}])

        assert result.content == "Async response"

    @pytest.mark.asyncio
    async def test_chat_async_routes_to_specified_backend(
        self, mock_lg: Logger
    ) -> None:
        """Test chat_async() routes to specified backend."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client(mock_lg, [response_a])
        client_b = make_client(mock_lg, [response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = await router.chat_async(
            [{"role": "user", "content": "Hi"}], backend="b"
        )

        assert result.content == "From B"

    @pytest.mark.asyncio
    async def test_chat_stream_async_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_stream_async() routes correctly."""
        response = ChatResponse(content="XYZ")
        client = make_client(mock_lg, [response])
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
        client_a = make_client(mock_lg, [response_a])
        client_b = make_client(mock_lg, [response_b])
        model_to_backend = {"model-a": "a", "model-b": "b"}
        router = LLMRouter(
            mock_lg, {"a": client_a, "b": client_b}, "a", model_to_backend
        )

        result = router.chat([{"role": "user", "content": "Hi"}], model="model-b")

        assert result.content == "From B"

    def test_falls_back_to_default_when_model_not_in_table(
        self, mock_lg: Logger
    ) -> None:
        """Test fallback to default when model not in routing table."""
        response_a = ChatResponse(content="From A")
        client_a = make_client(mock_lg, [response_a])
        model_to_backend = {"known-model": "a"}
        router = LLMRouter(mock_lg, {"a": client_a}, "a", model_to_backend)

        result = router.chat([{"role": "user", "content": "Hi"}], model="unknown-model")

        assert result.content == "From A"  # Falls back to default

    def test_explicit_backend_takes_priority_over_model(self, mock_lg: Logger) -> None:
        """Test explicit backend param overrides model-based routing."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client(mock_lg, [response_a])
        client_b = make_client(mock_lg, [response_b])
        model_to_backend = {"model-b": "b"}
        router = LLMRouter(
            mock_lg, {"a": client_a, "b": client_b}, "a", model_to_backend
        )

        # model="model-b" would route to b, but backend="a" overrides
        result = router.chat(
            [{"role": "user", "content": "Hi"}], model="model-b", backend="a"
        )

        assert result.content == "From A"

    def test_models_property_returns_routing_table(self, mock_lg: Logger) -> None:
        """Test models property exposes the routing table."""
        client = make_client(mock_lg)
        model_to_backend = {"model-x": "main", "model-y": "main"}
        router = LLMRouter(mock_lg, {"main": client}, "main", model_to_backend)

        assert router.models == {"model-x": "main", "model-y": "main"}

    def test_get_client_with_model_param(self, mock_lg: Logger) -> None:
        """Test get_client resolves by model."""
        client_a = make_client(mock_lg)
        client_b = make_client(mock_lg)
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
        client_a = LLMClient(lg=mock_lg, backend=backend_a)
        client_b = LLMClient(lg=mock_lg, backend=backend_b)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        router.close()

        assert backend_a._closed
        assert backend_b._closed

    @pytest.mark.asyncio
    async def test_aclose_closes_all_clients(self, mock_lg: Logger) -> None:
        """Test aclose() closes all clients."""
        backend_a = MockBackend()
        backend_b = MockBackend()
        client_a = LLMClient(lg=mock_lg, backend=backend_a)
        client_b = LLMClient(lg=mock_lg, backend=backend_b)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        await router.aclose()

        assert backend_a._aclosed
        assert backend_b._aclosed

    def test_sync_context_manager(self, mock_lg: Logger) -> None:
        """Test sync context manager calls close."""
        backend = MockBackend()
        client = LLMClient(lg=mock_lg, backend=backend)

        with LLMRouter(mock_lg, {"main": client}, "main") as router:
            assert router.clients == {"main": client}

        assert backend._closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_lg: Logger) -> None:
        """Test async context manager calls aclose."""
        backend = MockBackend()
        client = LLMClient(lg=mock_lg, backend=backend)

        async with LLMRouter(mock_lg, {"main": client}, "main") as router:
            assert router.clients == {"main": client}

        assert backend._aclosed


class TestLLMRouterCanCall:
    """Test LLMRouter.can_call() method."""

    def test_can_call_delegates_to_default_client(self, mock_lg: Logger) -> None:
        """Test can_call() delegates to default client."""
        backend = MockBackend()
        client = LLMClient(lg=mock_lg, backend=backend)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        # Client has no rate limiting, should return True
        assert router.can_call() is True

    def test_can_call_delegates_to_specified_backend(self, mock_lg: Logger) -> None:
        """Test can_call(backend=...) delegates to specified client."""
        from unittest.mock import MagicMock

        from appinfra.rate_limit import RateLimiter

        backend_a = MockBackend()
        backend_b = MockBackend()

        # Client A has rate limiting that returns False (rate limited)
        rate_limiter = MagicMock(spec=RateLimiter)
        rate_limiter.can_proceed.return_value = False

        client_a = LLMClient(lg=mock_lg, backend=backend_a, rate_limiter=rate_limiter)
        client_b = LLMClient(lg=mock_lg, backend=backend_b)  # No rate limiting

        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        # Default (a) should be rate limited
        assert router.can_call() is False
        # Backend b should be allowed
        assert router.can_call(backend="b") is True

    def test_can_call_with_model_routing(self, mock_lg: Logger) -> None:
        """Test can_call(model=...) uses model routing."""
        from unittest.mock import MagicMock

        from appinfra.rate_limit import RateLimiter

        backend_a = MockBackend()
        backend_b = MockBackend()

        # Client B has rate limiting that returns False (rate limited)
        rate_limiter = MagicMock(spec=RateLimiter)
        rate_limiter.can_proceed.return_value = False

        client_a = LLMClient(lg=mock_lg, backend=backend_a)
        client_b = LLMClient(lg=mock_lg, backend=backend_b, rate_limiter=rate_limiter)

        model_routing = {"model-a": "a", "model-b": "b"}
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a", model_routing)

        # Model-a routes to client_a (no rate limit)
        assert router.can_call(model="model-a") is True
        # Model-b routes to client_b (rate limited)
        assert router.can_call(model="model-b") is False

    def test_can_call_raises_on_unknown_backend(self, mock_lg: Logger) -> None:
        """Test can_call raises ValueError for unknown backend."""
        backend = MockBackend()
        client = LLMClient(lg=mock_lg, backend=backend)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            router.can_call(backend="unknown")


class TestLLMRouterResolve:
    """Test LLMRouter.resolve() method."""

    def test_resolve_returns_resolved_target(self, mock_lg: Logger) -> None:
        """Test resolve() returns a ResolvedTarget dataclass."""
        client = make_client(mock_lg)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.resolve()

        assert isinstance(result, ResolvedTarget)
        assert result.backend == "main"

    def test_resolve_uses_default_backend(self, mock_lg: Logger) -> None:
        """Test resolve() uses default backend when none specified."""
        client_a = make_client(mock_lg)
        client_b = make_client(mock_lg)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.resolve()

        assert result.backend == "a"

    def test_resolve_uses_explicit_backend(self, mock_lg: Logger) -> None:
        """Test resolve() uses explicit backend parameter."""
        client_a = make_client(mock_lg)
        client_b = make_client(mock_lg)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.resolve(backend="b")

        assert result.backend == "b"

    def test_resolve_routes_by_model(self, mock_lg: Logger) -> None:
        """Test resolve() routes by model when in routing table."""
        client_a = make_client(mock_lg)
        client_b = make_client(mock_lg)
        model_to_backend = {"gpt-4": "b"}
        router = LLMRouter(
            mock_lg, {"a": client_a, "b": client_b}, "a", model_to_backend
        )

        result = router.resolve(model="gpt-4")

        assert result.backend == "b"
        assert result.model == "gpt-4"

    def test_resolve_explicit_backend_overrides_model_routing(
        self, mock_lg: Logger
    ) -> None:
        """Test explicit backend takes priority over model-based routing."""
        client_a = make_client(mock_lg)
        client_b = make_client(mock_lg)
        model_to_backend = {"gpt-4": "b"}
        router = LLMRouter(
            mock_lg, {"a": client_a, "b": client_b}, "a", model_to_backend
        )

        result = router.resolve(model="gpt-4", backend="a")

        assert result.backend == "a"
        assert result.model == "gpt-4"

    def test_resolve_returns_explicit_model(self, mock_lg: Logger) -> None:
        """Test resolve() returns explicit model in result."""
        client = make_client(mock_lg)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.resolve(model="claude-3-opus")

        assert result.model == "claude-3-opus"

    def test_resolve_returns_client_default_model(self, mock_lg: Logger) -> None:
        """Test resolve() returns client's default_model when no model specified."""
        backend = MockBackend()
        client = LLMClient(lg=mock_lg, backend=backend, default_model="gpt-4-turbo")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.resolve()

        assert result.model == "gpt-4-turbo"

    def test_resolve_returns_none_model_when_no_default(self, mock_lg: Logger) -> None:
        """Test resolve() returns None model when client has no default."""
        client = make_client(mock_lg)  # No default_model
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.resolve()

        assert result.model is None

    def test_resolve_raises_on_unknown_backend(self, mock_lg: Logger) -> None:
        """Test resolve() raises ValueError for unknown backend."""
        client = make_client(mock_lg)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            router.resolve(backend="unknown")


class TestLLMClientDefaultModel:
    """Test LLMClient.default_model property."""

    def test_default_model_returns_configured_value(self, mock_lg: Logger) -> None:
        """Test default_model property returns the configured default."""
        backend = MockBackend()
        client = LLMClient(lg=mock_lg, backend=backend, default_model="gpt-4")

        assert client.default_model == "gpt-4"

    def test_default_model_returns_none_when_not_set(self, mock_lg: Logger) -> None:
        """Test default_model property returns None when not configured."""
        backend = MockBackend()
        client = LLMClient(lg=mock_lg, backend=backend)

        assert client.default_model is None
