"""Unit tests for lazy model discovery."""

from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import ModelConflictError, ModelDiscovery
from llm_infer.client.backends import Backend, BackendContext
from llm_infer.client.types import ChatRequest, ChatResponse, ResponseHolder

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


class MockBackend(Backend):
    """Mock backend with configurable model list."""

    def __init__(
        self,
        lg: Logger,
        name: str = "mock",
        models: list[str] | None = None,
        ctx: BackendContext | None = None,
        default_model: str | None = None,
    ) -> None:
        super().__init__(lg, name, ctx, default_model)
        self._models = models or []
        self._list_models_called = False
        self._last_response: ChatResponse | None = None

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    @property
    def provider(self) -> str:
        return "mock"

    def list_models(self) -> list[str]:
        self._list_models_called = True
        return self._models

    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="mock")

    def chat_stream(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> Iterator[str]:
        yield "mock"

    async def chat_async(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="mock")

    async def chat_stream_async(
        self, request: ChatRequest, holder: ResponseHolder | None = None
    ) -> AsyncIterator[str]:
        yield "mock"


class TestModelDiscovery:
    """Test ModelDiscovery class."""

    def test_loads_models_from_config(self, mock_lg: Logger) -> None:
        """Test that models from config are loaded immediately."""
        backend = MockBackend(mock_lg, models=["discovered-model"])

        discovery = ModelDiscovery(
            mock_lg,
            backends={"test": backend},
            configs={"test": {"models": ["config-model-1", "config-model-2"]}},
        )

        # Config models should be available immediately
        assert discovery.models == {
            "config-model-1": "test",
            "config-model-2": "test",
        }
        # Backend should NOT have been probed
        assert not backend._list_models_called

    def test_unknown_model_returns_none(self, mock_lg: Logger) -> None:
        """Test that unknown models return None without probing."""
        backend = MockBackend(mock_lg, models=["discovered-model"])

        discovery = ModelDiscovery(
            mock_lg,
            backends={"test": backend},
            configs={"test": {}},
        )

        # Request a model not in config - should return None, no probing
        result = discovery.get_backend_for_model("unknown-model")

        assert result is None
        assert not backend._list_models_called

    def test_no_probe_for_config_models(self, mock_lg: Logger) -> None:
        """Test that config models don't require backend probing."""
        backend = MockBackend(mock_lg, models=["other-model"])

        discovery = ModelDiscovery(
            mock_lg,
            backends={"test": backend},
            configs={"test": {"models": ["config-model"]}},
        )

        # Request a config model - should NOT trigger discovery
        result = discovery.get_backend_for_model("config-model")

        assert result == "test"
        assert not backend._list_models_called

    def test_get_models_for_backend_probes_lazily(self, mock_lg: Logger) -> None:
        """Test that get_models_for_backend probes on first call."""
        backend = MockBackend(mock_lg, models=["model-1", "model-2"])

        discovery = ModelDiscovery(
            mock_lg,
            backends={"test": backend},
            configs={"test": {}},
        )

        # First call should probe
        models = discovery.get_models_for_backend("test")
        assert backend._list_models_called
        assert models == ["model-1", "model-2"]

        # Reset call tracking
        backend._list_models_called = False

        # Second call should not probe again
        models = discovery.get_models_for_backend("test")
        assert not backend._list_models_called

    def test_config_model_conflict_raises(self, mock_lg: Logger) -> None:
        """Test that conflicting models in config raise ModelConflictError."""
        backend1 = MockBackend(mock_lg, name="backend1")
        backend2 = MockBackend(mock_lg, name="backend2")

        with pytest.raises(ModelConflictError) as exc_info:
            ModelDiscovery(
                mock_lg,
                backends={"backend1": backend1, "backend2": backend2},
                configs={
                    "backend1": {"models": ["shared"]},
                    "backend2": {"models": ["shared"]},
                },
            )

        assert exc_info.value.model == "shared"
        assert exc_info.value.backend1 == "backend1"
        assert exc_info.value.backend2 == "backend2"

    def test_discovered_backends_tracked(self, mock_lg: Logger) -> None:
        """Test that discovered backends are tracked."""
        backend = MockBackend(mock_lg, models=["model"])

        discovery = ModelDiscovery(
            mock_lg,
            backends={"test": backend},
            configs={"test": {}},
        )

        # Config models mark backend as discovered
        assert discovery.discovered_backends == set()

        discovery.get_models_for_backend("test")

        assert discovery.discovered_backends == {"test"}

    def test_discover_all(self, mock_lg: Logger) -> None:
        """Test discover_all probes all backends."""
        backend1 = MockBackend(mock_lg, name="backend1", models=["model-1"])
        backend2 = MockBackend(mock_lg, name="backend2", models=["model-2"])

        discovery = ModelDiscovery(
            mock_lg,
            backends={"backend1": backend1, "backend2": backend2},
            configs={"backend1": {}, "backend2": {}},
        )

        models = discovery.discover_all()

        assert backend1._list_models_called
        assert backend2._list_models_called
        assert models == {"model-1": "backend1", "model-2": "backend2"}
