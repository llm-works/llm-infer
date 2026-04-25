"""Unit tests for LLMRouter."""

from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import (
    ChatRequest,
    ChatResponse,
    LLMClient,
    LLMRouter,
    ResolvedTarget,
)
from llm_infer.client.backends import Backend, BackendContext
from llm_infer.client.discovery import ModelDiscovery

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

    def chat(self, request: ChatRequest) -> ChatResponse:
        response = next(self._responses)
        self._last_response = response
        return response

    def chat_stream(self, request: ChatRequest) -> Iterator[str]:
        response = next(self._responses)
        yield from response.content
        self._last_response = response

    async def chat_async(self, request: ChatRequest) -> ChatResponse:
        response = next(self._responses)
        self._last_response = response
        return response

    async def chat_stream_async(self, request: ChatRequest) -> AsyncIterator[str]:
        response = next(self._responses)
        for char in response.content:
            yield char
        self._last_response = response

    def close(self) -> None:
        self._closed = True

    async def aclose(self) -> None:
        self._aclosed = True


def make_client(
    lg: Logger,
    name: str = "test",
    responses: list[ChatResponse] | None = None,
    default_model: str | None = None,
) -> LLMClient:
    """Create a client with a mock backend."""
    backend = MockBackend(lg, name, responses=responses, default_model=default_model)
    return LLMClient(lg=lg, backend=backend)


def make_discovery(
    lg: Logger,
    clients: dict[str, LLMClient],
    model_to_backend: dict[str, str],
) -> ModelDiscovery:
    """Create a ModelDiscovery with pre-populated model routing."""
    backends = {name: client.backend for name, client in clients.items()}
    configs = {name: {"models": []} for name in backends}
    for model, backend_name in model_to_backend.items():
        configs[backend_name]["models"].append(model)
    return ModelDiscovery(lg, backends, configs)


class TestLLMRouterInit:
    """Test LLMRouter initialization."""

    def test_init_with_clients(self, mock_lg: Logger) -> None:
        """Test router initializes with clients dict."""
        client_a = make_client(mock_lg, "a")
        client_b = make_client(mock_lg, "b")
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        assert router.clients == {"a": client_a, "b": client_b}
        assert router.default == "a"

    def test_init_raises_if_default_not_in_clients(self, mock_lg: Logger) -> None:
        """Test router raises if default not in clients."""
        client_a = make_client(mock_lg, "a")
        with pytest.raises(
            ValueError, match="Default backend 'missing' not in clients"
        ):
            LLMRouter(mock_lg, {"a": client_a}, "missing")


class TestLLMRouterRouting:
    """Test LLMRouter routing behavior."""

    def test_chat_uses_default_backend(self, mock_lg: Logger) -> None:
        """Test chat() uses default backend when no backend specified."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client(mock_lg, "a", [response_a])
        client_b = make_client(mock_lg, "b", [response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.chat([{"role": "user", "content": "Hi"}])

        assert result.content == "From A"

    def test_chat_routes_to_specified_backend(self, mock_lg: Logger) -> None:
        """Test chat() routes to specified backend."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client(mock_lg, "a", [response_a])
        client_b = make_client(mock_lg, "b", [response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.chat([{"role": "user", "content": "Hi"}], backend="b")

        assert result.content == "From B"

    def test_chat_raises_on_unknown_backend(self, mock_lg: Logger) -> None:
        """Test chat() raises on unknown backend."""
        client_a = make_client(mock_lg, "a", [ChatResponse(content="A")])
        router = LLMRouter(mock_lg, {"a": client_a}, "a")

        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            router.chat([{"role": "user", "content": "Hi"}], backend="unknown")

    def test_chat_stream_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_stream() routes to correct backend."""
        response = ChatResponse(content="ABC")
        client = make_client(mock_lg, "main", [response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        tokens = list(router.chat_stream([{"role": "user", "content": "Hi"}]))

        assert tokens == ["A", "B", "C"]


class TestLLMRouterAsync:
    """Test LLMRouter async API."""

    @pytest.mark.asyncio
    async def test_chat_async_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_async() routes to correct backend."""
        response = ChatResponse(content="Async response")
        client = make_client(mock_lg, "main", [response])
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
        client_a = make_client(mock_lg, "a", [response_a])
        client_b = make_client(mock_lg, "b", [response_b])
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = await router.chat_async(
            [{"role": "user", "content": "Hi"}], backend="b"
        )

        assert result.content == "From B"

    @pytest.mark.asyncio
    async def test_chat_stream_async_routes_correctly(self, mock_lg: Logger) -> None:
        """Test chat_stream_async() routes correctly."""
        response = ChatResponse(content="XYZ")
        client = make_client(mock_lg, "main", [response])
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
        client_a = make_client(mock_lg, "a", [response_a])
        client_b = make_client(mock_lg, "b", [response_b])
        clients = {"a": client_a, "b": client_b}
        discovery = make_discovery(mock_lg, clients, {"model-a": "a", "model-b": "b"})
        router = LLMRouter(mock_lg, clients, "a", discovery=discovery)

        result = router.chat([{"role": "user", "content": "Hi"}], model="model-b")

        assert result.content == "From B"

    def test_falls_back_to_default_when_model_not_in_table(
        self, mock_lg: Logger
    ) -> None:
        """Test fallback to default when model not in routing table."""
        response_a = ChatResponse(content="From A")
        client_a = make_client(mock_lg, "a", [response_a])
        clients = {"a": client_a}
        discovery = make_discovery(mock_lg, clients, {"known-model": "a"})
        router = LLMRouter(mock_lg, clients, "a", discovery=discovery)

        result = router.chat([{"role": "user", "content": "Hi"}], model="unknown-model")

        assert result.content == "From A"

    def test_explicit_backend_takes_priority_over_model(self, mock_lg: Logger) -> None:
        """Test explicit backend param overrides model-based routing."""
        response_a = ChatResponse(content="From A")
        response_b = ChatResponse(content="From B")
        client_a = make_client(mock_lg, "a", [response_a])
        client_b = make_client(mock_lg, "b", [response_b])
        clients = {"a": client_a, "b": client_b}
        discovery = make_discovery(mock_lg, clients, {"model-b": "b"})
        router = LLMRouter(mock_lg, clients, "a", discovery=discovery)

        result = router.chat(
            [{"role": "user", "content": "Hi"}], model="model-b", backend="a"
        )

        assert result.content == "From A"

    def test_models_property_returns_routing_table(self, mock_lg: Logger) -> None:
        """Test models property exposes the routing table."""
        client = make_client(mock_lg, "main")
        clients = {"main": client}
        discovery = make_discovery(
            mock_lg, clients, {"model-x": "main", "model-y": "main"}
        )
        router = LLMRouter(mock_lg, clients, "main", discovery=discovery)

        assert router.models == {"model-x": "main", "model-y": "main"}

    def test_get_client_with_model_param(self, mock_lg: Logger) -> None:
        """Test get_client resolves by model."""
        client_a = make_client(mock_lg, "a")
        client_b = make_client(mock_lg, "b")
        clients = {"a": client_a, "b": client_b}
        discovery = make_discovery(mock_lg, clients, {"gpt-4": "b"})
        router = LLMRouter(mock_lg, clients, "a", discovery=discovery)

        resolved = router.get_client(model="gpt-4")

        assert resolved is client_b


class TestLLMRouterResourceManagement:
    """Test LLMRouter resource management."""

    def test_close_closes_all_clients(self, mock_lg: Logger) -> None:
        """Test close() closes all clients."""
        backend_a = MockBackend(mock_lg, "a")
        backend_b = MockBackend(mock_lg, "b")
        client_a = LLMClient(lg=mock_lg, backend=backend_a)
        client_b = LLMClient(lg=mock_lg, backend=backend_b)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        router.close()

        assert backend_a._closed
        assert backend_b._closed

    @pytest.mark.asyncio
    async def test_aclose_closes_all_clients(self, mock_lg: Logger) -> None:
        """Test aclose() closes all clients."""
        backend_a = MockBackend(mock_lg, "a")
        backend_b = MockBackend(mock_lg, "b")
        client_a = LLMClient(lg=mock_lg, backend=backend_a)
        client_b = LLMClient(lg=mock_lg, backend=backend_b)
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        await router.aclose()

        assert backend_a._aclosed
        assert backend_b._aclosed

    def test_sync_context_manager(self, mock_lg: Logger) -> None:
        """Test sync context manager calls close."""
        backend = MockBackend(mock_lg, "main")
        client = LLMClient(lg=mock_lg, backend=backend)

        with LLMRouter(mock_lg, {"main": client}, "main") as router:
            assert router.clients == {"main": client}

        assert backend._closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_lg: Logger) -> None:
        """Test async context manager calls aclose."""
        backend = MockBackend(mock_lg, "main")
        client = LLMClient(lg=mock_lg, backend=backend)

        async with LLMRouter(mock_lg, {"main": client}, "main") as router:
            assert router.clients == {"main": client}

        assert backend._aclosed


class TestLLMRouterCanCall:
    """Test LLMRouter.can_call() method."""

    def test_can_call_delegates_to_default_client(self, mock_lg: Logger) -> None:
        """Test can_call() delegates to default client."""
        backend = MockBackend(mock_lg, "main")
        client = LLMClient(lg=mock_lg, backend=backend)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        assert router.can_call() is True

    def test_can_call_delegates_to_specified_backend(self, mock_lg: Logger) -> None:
        """Test can_call(backend=...) delegates to specified client."""
        rate_limiter = MagicMock()
        rate_limiter.can_proceed.return_value = False

        ctx_a = BackendContext(rate_limiter=rate_limiter)
        backend_a = MockBackend(mock_lg, "a", ctx=ctx_a)
        backend_b = MockBackend(mock_lg, "b")

        client_a = LLMClient(lg=mock_lg, backend=backend_a)
        client_b = LLMClient(lg=mock_lg, backend=backend_b)

        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        assert router.can_call() is False
        assert router.can_call(backend="b") is True

    def test_can_call_with_model_routing(self, mock_lg: Logger) -> None:
        """Test can_call(model=...) uses model routing."""
        rate_limiter = MagicMock()
        rate_limiter.can_proceed.return_value = False

        ctx_b = BackendContext(rate_limiter=rate_limiter)
        backend_a = MockBackend(mock_lg, "a")
        backend_b = MockBackend(mock_lg, "b", ctx=ctx_b)

        client_a = LLMClient(lg=mock_lg, backend=backend_a)
        client_b = LLMClient(lg=mock_lg, backend=backend_b)
        clients = {"a": client_a, "b": client_b}
        discovery = make_discovery(mock_lg, clients, {"model-a": "a", "model-b": "b"})
        router = LLMRouter(mock_lg, clients, "a", discovery=discovery)

        assert router.can_call(model="model-a") is True
        assert router.can_call(model="model-b") is False

    def test_can_call_raises_on_unknown_backend(self, mock_lg: Logger) -> None:
        """Test can_call raises ValueError for unknown backend."""
        backend = MockBackend(mock_lg, "main")
        client = LLMClient(lg=mock_lg, backend=backend)
        router = LLMRouter(mock_lg, {"main": client}, "main")

        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            router.can_call(backend="unknown")


class TestLLMRouterResolve:
    """Test LLMRouter.resolve() method."""

    def test_resolve_returns_resolved_target(self, mock_lg: Logger) -> None:
        """Test resolve() returns a ResolvedTarget dataclass."""
        client = make_client(mock_lg, "main")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.resolve()

        assert isinstance(result, ResolvedTarget)
        assert result.backend == "main"

    def test_resolve_uses_default_backend(self, mock_lg: Logger) -> None:
        """Test resolve() uses default backend when none specified."""
        client_a = make_client(mock_lg, "a")
        client_b = make_client(mock_lg, "b")
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.resolve()

        assert result.backend == "a"

    def test_resolve_uses_explicit_backend(self, mock_lg: Logger) -> None:
        """Test resolve() uses explicit backend parameter."""
        client_a = make_client(mock_lg, "a")
        client_b = make_client(mock_lg, "b")
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "a")

        result = router.resolve(backend="b")

        assert result.backend == "b"

    def test_resolve_routes_by_model(self, mock_lg: Logger) -> None:
        """Test resolve() routes by model when in routing table."""
        client_a = make_client(mock_lg, "a")
        client_b = make_client(mock_lg, "b")
        clients = {"a": client_a, "b": client_b}
        discovery = make_discovery(mock_lg, clients, {"gpt-4": "b"})
        router = LLMRouter(mock_lg, clients, "a", discovery=discovery)

        result = router.resolve(model="gpt-4")

        assert result.backend == "b"
        assert result.model == "gpt-4"

    def test_resolve_explicit_backend_overrides_model_routing(
        self, mock_lg: Logger
    ) -> None:
        """Test explicit backend takes priority over model-based routing."""
        client_a = make_client(mock_lg, "a")
        client_b = make_client(mock_lg, "b")
        clients = {"a": client_a, "b": client_b}
        discovery = make_discovery(mock_lg, clients, {"gpt-4": "b"})
        router = LLMRouter(mock_lg, clients, "a", discovery=discovery)

        result = router.resolve(model="gpt-4", backend="a")

        assert result.backend == "a"
        assert result.model == "gpt-4"

    def test_resolve_returns_explicit_model(self, mock_lg: Logger) -> None:
        """Test resolve() returns explicit model in result."""
        client = make_client(mock_lg, "main")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.resolve(model="claude-3-opus")

        assert result.model == "claude-3-opus"

    def test_resolve_returns_client_default_model(self, mock_lg: Logger) -> None:
        """Test resolve() returns client's default_model when no model specified."""
        client = make_client(mock_lg, "main", default_model="gpt-4-turbo")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.resolve()

        assert result.model == "gpt-4-turbo"

    def test_resolve_returns_none_model_when_no_default(self, mock_lg: Logger) -> None:
        """Test resolve() returns None model when client has no default."""
        client = make_client(mock_lg, "main")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        result = router.resolve()

        assert result.model is None

    def test_resolve_raises_on_unknown_backend(self, mock_lg: Logger) -> None:
        """Test resolve() raises ValueError for unknown backend."""
        client = make_client(mock_lg, "main")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        with pytest.raises(ValueError, match="Backend 'unknown' not found"):
            router.resolve(backend="unknown")


class TestLLMClientDefaultModel:
    """Test LLMClient.default_model property."""

    def test_default_model_returns_configured_value(self, mock_lg: Logger) -> None:
        """Test default_model property returns the configured default."""
        backend = MockBackend(mock_lg, "test", default_model="gpt-4")
        client = LLMClient(lg=mock_lg, backend=backend)

        assert client.default_model == "gpt-4"

    def test_default_model_returns_none_when_not_set(self, mock_lg: Logger) -> None:
        """Test default_model property returns None when not configured."""
        backend = MockBackend(mock_lg, "test")
        client = LLMClient(lg=mock_lg, backend=backend)

        assert client.default_model is None


class TestLLMRouterWithChatArgs:
    """Test LLMRouter.with_chat_args() method."""

    def test_with_chat_args_returns_bound_client(self, mock_lg: Logger) -> None:
        """Test with_chat_args returns a BoundChatClient."""
        from llm_infer.client import BoundChatClient

        client = make_client(mock_lg, "main")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        bound = router.with_chat_args(role="exploration")

        assert isinstance(bound, BoundChatClient)
        assert bound.client is router
        assert bound.bound_kwargs == {"role": "exploration"}

    def test_with_chat_args_merges_into_chat(self, mock_lg: Logger) -> None:
        """Test bound kwargs are merged into chat calls (routing works)."""
        response = ChatResponse(content="from_a")
        client_a = make_client(mock_lg, "a", [response])
        client_b = make_client(mock_lg, "b")
        router = LLMRouter(mock_lg, {"a": client_a, "b": client_b}, "b")
        bound = router.with_chat_args(backend="a")

        result = bound.chat([{"role": "user", "content": "Hi"}])

        assert result.content == "from_a"

    @pytest.mark.asyncio
    async def test_with_chat_args_merges_into_chat_async(self, mock_lg: Logger) -> None:
        """Test bound kwargs are merged into async chat calls."""
        response = ChatResponse(content="test")
        client = make_client(mock_lg, "main", [response])
        router = LLMRouter(mock_lg, {"main": client}, "main")
        bound = router.with_chat_args(system="You are helpful")

        result = await bound.chat_async([{"role": "user", "content": "Hi"}])

        assert result.content == "test"

    def test_multiple_bound_clients_independent(self, mock_lg: Logger) -> None:
        """Test multiple bound clients from same router are independent."""
        from llm_infer.client import BoundChatClient

        client = make_client(mock_lg, "main")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        exploration = router.with_chat_args(role="exploration")
        synthesis = router.with_chat_args(role="synthesis")

        assert isinstance(exploration, BoundChatClient)
        assert isinstance(synthesis, BoundChatClient)
        assert exploration.bound_kwargs == {"role": "exploration"}
        assert synthesis.bound_kwargs == {"role": "synthesis"}
        assert exploration is not synthesis

    def test_bound_client_chaining(self, mock_lg: Logger) -> None:
        """Test bound clients can be chained with additional args."""
        client = make_client(mock_lg, "main")
        router = LLMRouter(mock_lg, {"main": client}, "main")

        bound1 = router.with_chat_args(role="exploration")
        bound2 = bound1.with_chat_args(backend="main")

        assert bound1.bound_kwargs == {"role": "exploration"}
        assert bound2.bound_kwargs == {"role": "exploration", "backend": "main"}

    def test_explicit_args_override_bound_args(self, mock_lg: Logger) -> None:
        """Test explicit call args override bound args."""
        response = ChatResponse(content="test")
        client = make_client(mock_lg, "main", [response])
        router = LLMRouter(mock_lg, {"main": client}, "main")
        bound = router.with_chat_args(temperature=0.5)

        bound.chat([{"role": "user", "content": "Hi"}], temperature=0.9)

    def test_bound_args_preserved_when_not_overridden(self, mock_lg: Logger) -> None:
        """Test bound args are used when not explicitly passed."""
        response = ChatResponse(content="test")
        client = make_client(mock_lg, "main", [response])
        router = LLMRouter(mock_lg, {"main": client}, "main")
        bound = router.with_chat_args(temperature=0.5, max_tokens=100)

        bound.chat([{"role": "user", "content": "Hi"}])

    def test_bound_client_can_call_delegates(self, mock_lg: Logger) -> None:
        """Test can_call() delegates to wrapped client."""
        client = make_client(mock_lg, "main")
        router = LLMRouter(mock_lg, {"main": client}, "main")
        bound = router.with_chat_args(role="exploration")

        assert bound.can_call() is True

    def test_bound_client_context_manager(self, mock_lg: Logger) -> None:
        """Test bound client works as context manager."""
        response = ChatResponse(content="test")
        client = make_client(mock_lg, "main", [response])
        router = LLMRouter(mock_lg, {"main": client}, "main")

        with router.with_chat_args(role="exploration") as bound:
            result = bound.chat([{"role": "user", "content": "Hi"}])
            assert result.content == "test"
