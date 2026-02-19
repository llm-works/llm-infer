"""Unit tests for lazy model discovery."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import Factory, LLMClient, ModelConflictError, ModelDiscovery
from llm_infer.client.backends import Backend
from llm_infer.client.types import ChatResponse

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


class MockBackend(Backend):
    """Mock backend with configurable model list."""

    def __init__(self, models: list[str] | None = None) -> None:
        self._models = models or []
        self._last_response: ChatResponse | None = None
        self._list_models_called = False

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    def list_models(self) -> list[str]:
        self._list_models_called = True
        return self._models

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content="mock")

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
        yield "mock"

    async def chat_async(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(content="mock")

    async def chat_stream_async(self, messages: list[dict[str, Any]], **kwargs: Any):
        yield "mock"

    @classmethod
    def from_config(cls, lg: Logger, config: dict[str, Any]) -> "MockBackend":
        return cls(models=config.get("_test_models", []))


class TestModelDiscovery:
    """Test ModelDiscovery class."""

    def test_loads_models_from_config(self, mock_lg: Logger) -> None:
        """Test that models from config are loaded immediately."""
        backend = MockBackend(models=["discovered-model"])
        client = LLMClient(lg=mock_lg, backend=backend)

        discovery = ModelDiscovery(
            mock_lg,
            clients={"test": client},
            configs={"test": {"models": ["config-model-1", "config-model-2"]}},
        )

        # Config models should be available immediately
        assert discovery.models == {
            "config-model-1": "test",
            "config-model-2": "test",
        }
        # Backend should NOT have been probed
        assert not backend._list_models_called

    def test_lazy_probe_disabled(self, mock_lg: Logger) -> None:
        """Test that lazy_probe=False prevents backend probing."""
        backend = MockBackend(models=["discovered-model"])
        client = LLMClient(lg=mock_lg, backend=backend)

        discovery = ModelDiscovery(
            mock_lg,
            clients={"test": client},
            configs={"test": {}},
            lazy_probe=False,
        )

        # Request a model not in config
        result = discovery.get_backend_for_model("discovered-model")

        # Should return None (no probing)
        assert result is None
        assert not backend._list_models_called

    def test_unknown_model_returns_none(self, mock_lg: Logger) -> None:
        """Test that unknown models return None without probing."""
        backend = MockBackend(models=["discovered-model"])
        client = LLMClient(lg=mock_lg, backend=backend)

        discovery = ModelDiscovery(
            mock_lg,
            clients={"test": client},
            configs={"test": {}},
        )

        # Request a model not in config - should return None, no probing
        result = discovery.get_backend_for_model("unknown-model")

        assert result is None
        assert not backend._list_models_called

    def test_no_probe_for_config_models(self, mock_lg: Logger) -> None:
        """Test that config models don't require backend probing."""
        backend = MockBackend(models=["other-model"])
        client = LLMClient(lg=mock_lg, backend=backend)

        discovery = ModelDiscovery(
            mock_lg,
            clients={"test": client},
            configs={"test": {"models": ["config-model"]}},
        )

        # Request a config model - should NOT trigger discovery
        result = discovery.get_backend_for_model("config-model")

        assert result == "test"
        assert not backend._list_models_called

    def test_explicit_discover_only_probes_once(self, mock_lg: Logger) -> None:
        """Test that discover_backend() only probes once per backend."""
        backend = MockBackend(models=["model-1"])
        client = LLMClient(lg=mock_lg, backend=backend)

        discovery = ModelDiscovery(
            mock_lg,
            clients={"test": client},
            configs={"test": {}},
        )

        # First explicit discovery
        discovery.discover_backend("test")
        assert backend._list_models_called
        assert "model-1" in discovery.models

        # Reset call tracking
        backend._list_models_called = False

        # Second call should not probe again
        discovery.discover_backend("test")
        assert not backend._list_models_called

    def test_config_model_conflict_raises(self, mock_lg: Logger) -> None:
        """Test that conflicting models in config raise ModelConflictError."""
        backend1 = MockBackend()
        backend2 = MockBackend()
        client1 = LLMClient(lg=mock_lg, backend=backend1)
        client2 = LLMClient(lg=mock_lg, backend=backend2)

        with pytest.raises(ModelConflictError) as exc_info:
            ModelDiscovery(
                mock_lg,
                clients={"backend1": client1, "backend2": client2},
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
        backend = MockBackend(models=["model"])
        client = LLMClient(lg=mock_lg, backend=backend)

        discovery = ModelDiscovery(
            mock_lg,
            clients={"test": client},
            configs={"test": {}},
        )

        assert discovery.discovered_backends == set()

        discovery.discover_backend("test")

        assert discovery.discovered_backends == {"test"}

    def test_discover_all(self, mock_lg: Logger) -> None:
        """Test discover_all probes all backends."""
        backend1 = MockBackend(models=["model-1"])
        backend2 = MockBackend(models=["model-2"])
        client1 = LLMClient(lg=mock_lg, backend=backend1)
        client2 = LLMClient(lg=mock_lg, backend=backend2)

        discovery = ModelDiscovery(
            mock_lg,
            clients={"backend1": client1, "backend2": client2},
            configs={"backend1": {}, "backend2": {}},
        )

        models = discovery.discover_all()

        assert backend1._list_models_called
        assert backend2._list_models_called
        assert models == {"model-1": "backend1", "model-2": "backend2"}


class TestFactoryLazyDiscovery:
    """Test Factory integration with lazy discovery."""

    def test_no_probing_at_startup(self, mock_lg: Logger) -> None:
        """Test that Factory.from_config doesn't probe backends at startup."""
        # Register mock backend
        Factory.register("mock", MockBackend)

        factory = Factory(mock_lg)
        config = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "mock",
                    "_test_models": ["local-model"],
                },
                "remote": {
                    "type": "mock",
                    "_test_models": ["remote-model"],
                },
            },
        }

        router = factory.from_config(config)

        # Neither backend should have been probed
        local_backend: MockBackend = router.clients["local"].backend  # type: ignore
        remote_backend: MockBackend = router.clients["remote"].backend  # type: ignore
        assert not local_backend._list_models_called
        assert not remote_backend._list_models_called

        router.close()

    def test_auto_model_probes_default_backend(self, mock_lg: Logger) -> None:
        """Test that model='auto' probes only the default backend."""
        Factory.register("mock", MockBackend)

        factory = Factory(mock_lg)
        config = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "mock",
                    "_test_models": ["local-model"],
                },
                "remote": {
                    "type": "mock",
                    "_test_models": ["remote-model"],
                },
            },
        }

        router = factory.from_config(config)

        # model="auto" should route to default and probe it
        resolved = router.resolve(model="auto")

        local_backend: MockBackend = router.clients["local"].backend  # type: ignore
        remote_backend: MockBackend = router.clients["remote"].backend  # type: ignore

        assert resolved.backend == "local"
        assert resolved.model == "local-model"  # Resolved to actual model
        assert local_backend._list_models_called
        assert not remote_backend._list_models_called  # Remote not probed

        router.close()

    def test_explicit_backend_skips_discovery(self, mock_lg: Logger) -> None:
        """Test that explicit backend= doesn't trigger discovery."""
        Factory.register("mock", MockBackend)

        factory = Factory(mock_lg)
        config = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "mock",
                    "_test_models": ["local-model"],
                },
                "remote": {
                    "type": "mock",
                    "_test_models": ["remote-model"],
                },
            },
        }

        router = factory.from_config(config)

        # Explicit backend - should not trigger discovery
        resolved = router.resolve(backend="remote")

        local_backend: MockBackend = router.clients["local"].backend  # type: ignore
        remote_backend: MockBackend = router.clients["remote"].backend  # type: ignore

        assert resolved.backend == "remote"
        assert not local_backend._list_models_called
        assert not remote_backend._list_models_called

        router.close()

    def test_config_models_available_without_probing(self, mock_lg: Logger) -> None:
        """Test that config-specified models work without probing."""
        Factory.register("mock", MockBackend)

        factory = Factory(mock_lg)
        config = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "mock",
                    "models": ["explicitly-listed"],  # Config-specified
                    "_test_models": ["discovered-model"],
                },
            },
        }

        router = factory.from_config(config)

        # Config model should resolve without probing
        resolved = router.resolve(model="explicitly-listed")

        local_backend: MockBackend = router.clients["local"].backend  # type: ignore
        assert resolved.backend == "local"
        assert not local_backend._list_models_called

        router.close()

    def test_default_model_uses_backend_default(self, mock_lg: Logger) -> None:
        """Test that model='default' uses the backend's configured default_model."""
        Factory.register("mock", MockBackend)

        factory = Factory(mock_lg)
        config = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "mock",
                    "model": "my-default-model",  # Configured default
                    "_test_models": ["my-default-model", "other-model"],
                },
            },
        }

        router = factory.from_config(config)

        # model="default" should resolve to configured default_model
        resolved = router.resolve(model="default")

        local_backend: MockBackend = router.clients["local"].backend  # type: ignore
        assert resolved.backend == "local"
        assert resolved.model == "my-default-model"
        assert not local_backend._list_models_called  # No probing needed

        router.close()

    def test_unknown_model_falls_back_to_default_backend(self, mock_lg: Logger) -> None:
        """Test that unknown models route to default backend without probing."""
        Factory.register("mock", MockBackend)

        factory = Factory(mock_lg)
        config = {
            "default": "local",
            "backends": {
                "local": {
                    "type": "mock",
                    "_test_models": ["local-model"],
                },
                "remote": {
                    "type": "mock",
                    "_test_models": ["remote-model"],
                },
            },
        }

        router = factory.from_config(config)

        # Unknown model should route to default without probing any backend
        resolved = router.resolve(model="unknown-model")

        local_backend: MockBackend = router.clients["local"].backend  # type: ignore
        remote_backend: MockBackend = router.clients["remote"].backend  # type: ignore

        assert resolved.backend == "local"
        assert resolved.model == "unknown-model"  # Passed through
        assert not local_backend._list_models_called
        assert not remote_backend._list_models_called

        router.close()
