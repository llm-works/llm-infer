"""Unit tests for backend registry."""

import pytest
from appinfra.log import create_lg

from llm_infer.backends.linear.formats.base import QuantFormat
from llm_infer.backends.linear.registry import BackendRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def lg():
    """Create a logger for tests."""
    return create_lg("test", "debug")


@pytest.fixture
def registry(lg):
    """Create a BackendRegistry for tests."""
    return BackendRegistry(lg)


class TestBackendRegistryGet:
    """Test BackendRegistry.get method."""

    def test_get_awq_backend_returns_backend(self, registry) -> None:
        """Test that we can get an AWQ backend."""
        backend = registry.get(QuantFormat.AWQ)
        assert backend is not None
        assert backend.format == QuantFormat.AWQ

    def test_get_fp8_backend_returns_backend(self, registry) -> None:
        """Test that we can get an FP8 backend."""
        backend = registry.get(QuantFormat.FP8)
        assert backend is not None
        assert backend.format == QuantFormat.FP8

    def test_get_with_pytorch_preference(self, registry) -> None:
        """Test that we can request pytorch backend specifically."""
        backend = registry.get(QuantFormat.AWQ, preference="pytorch")
        assert backend is not None
        assert backend.name == "pytorch"

    def test_get_with_invalid_preference_falls_back(self, registry) -> None:
        """Test that invalid preference falls back to auto-selection."""
        backend = registry.get(QuantFormat.AWQ, preference="nonexistent")
        assert backend is not None  # Should fall back to available backend


class TestBackendRegistryListAvailable:
    """Test BackendRegistry.list_available method."""

    def test_awq_has_available_backends(self, registry) -> None:
        """Test that AWQ has at least one available backend."""
        backends = registry.list_available(QuantFormat.AWQ)
        assert len(backends) >= 1
        assert "pytorch" in backends  # PyTorch backend always available

    def test_fp8_has_available_backends(self, registry) -> None:
        """Test that FP8 has at least one available backend."""
        backends = registry.list_available(QuantFormat.FP8)
        assert len(backends) >= 1
        assert "pytorch" in backends  # PyTorch backend always available


class TestBackendRegistryErrors:
    """Test error handling in BackendRegistry."""

    def test_get_unsupported_format_raises(self, registry) -> None:
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="No backends defined"):
            registry.get(QuantFormat.NONE)

    def test_list_available_none_format_returns_empty(self, registry) -> None:
        """Test that NONE format returns empty list."""
        backends = registry.list_available(QuantFormat.NONE)
        assert backends == []
