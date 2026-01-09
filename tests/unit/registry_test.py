"""Unit tests for backend registry."""

import pytest

from llm_infer.backends.linear.formats.base import QuantFormat
from llm_infer.backends.linear.registry import (
    get_available_backends,
    get_backend,
    get_linear_backend,
)

pytestmark = pytest.mark.unit


class TestGetBackend:
    """Test get_backend function."""

    def test_get_awq_backend_returns_backend(self) -> None:
        """Test that we can get an AWQ backend."""
        backend = get_backend(QuantFormat.AWQ)
        assert backend is not None
        assert backend.format == QuantFormat.AWQ

    def test_get_fp8_backend_returns_backend(self) -> None:
        """Test that we can get an FP8 backend."""
        backend = get_backend(QuantFormat.FP8)
        assert backend is not None
        assert backend.format == QuantFormat.FP8

    def test_get_backend_with_pytorch_preference(self) -> None:
        """Test that we can request pytorch backend specifically."""
        backend = get_backend(QuantFormat.AWQ, preference="pytorch")
        assert backend is not None
        assert backend.name == "pytorch"

    def test_get_backend_with_invalid_preference_falls_back(self) -> None:
        """Test that invalid preference falls back to auto-selection."""
        backend = get_backend(QuantFormat.AWQ, preference="nonexistent")
        assert backend is not None  # Should fall back to available backend


class TestGetAvailableBackends:
    """Test get_available_backends function."""

    def test_awq_has_available_backends(self) -> None:
        """Test that AWQ has at least one available backend."""
        backends = get_available_backends(QuantFormat.AWQ)
        assert len(backends) >= 1
        assert "pytorch" in backends  # PyTorch backend always available

    def test_fp8_has_available_backends(self) -> None:
        """Test that FP8 has at least one available backend."""
        backends = get_available_backends(QuantFormat.FP8)
        assert len(backends) >= 1
        assert "pytorch" in backends  # PyTorch backend always available


class TestGetLinearBackend:
    """Test backward compatibility get_linear_backend function."""

    def test_auto_returns_backend(self) -> None:
        """Test that 'auto' returns a backend."""
        backend = get_linear_backend("auto")
        assert backend is not None
        assert backend.format == QuantFormat.AWQ

    def test_pytorch_returns_pytorch_backend(self) -> None:
        """Test that 'pytorch' returns the pytorch backend."""
        backend = get_linear_backend("pytorch")
        assert backend.name == "pytorch"

    def test_default_is_auto(self) -> None:
        """Test that default argument is 'auto'."""
        backend = get_linear_backend()
        assert backend is not None


class TestGetBackendErrors:
    """Test error handling in get_backend."""

    def test_get_backend_unregistered_format_raises(self) -> None:
        """Test that unregistered format raises ValueError."""
        with pytest.raises(ValueError, match="No backends registered"):
            get_backend(QuantFormat.NONE)


class TestGetAvailableBackendsEdgeCases:
    """Test edge cases for get_available_backends."""

    def test_none_format_returns_empty(self) -> None:
        """Test that NONE format returns empty list."""
        backends = get_available_backends(QuantFormat.NONE)
        assert backends == []
