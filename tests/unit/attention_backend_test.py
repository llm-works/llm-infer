"""Unit tests for attention backend selection."""

import pytest

from llm_infer.engines.native.attention import (
    FLASHINFER_AVAILABLE,
    NaiveAttentionBackend,
    get_attention_backend,
)

pytestmark = pytest.mark.unit


class TestGetAttentionBackend:
    """Test get_attention_backend function."""

    def test_naive_preference(self) -> None:
        """Test that naive preference returns NaiveAttentionBackend."""
        backend = get_attention_backend("naive")
        assert isinstance(backend, NaiveAttentionBackend)

    def test_auto_returns_backend(self) -> None:
        """Test that auto preference returns a valid backend."""
        backend = get_attention_backend("auto")
        # Should return either FlashInfer or Naive depending on availability
        assert backend is not None

    def test_invalid_preference_raises(self) -> None:
        """Test that invalid preference raises ValueError."""
        with pytest.raises(ValueError, match="Unknown attention backend"):
            get_attention_backend("invalid_backend")

    def test_flashinfer_unavailable_raises(self) -> None:
        """Test that requesting flashinfer when unavailable raises RuntimeError."""
        if FLASHINFER_AVAILABLE:
            pytest.skip("FlashInfer is available")
        with pytest.raises(RuntimeError, match="FlashInfer backend requested"):
            get_attention_backend("flashinfer")


class TestNaiveAttentionBackend:
    """Test NaiveAttentionBackend initialization."""

    def test_create_backend(self) -> None:
        """Test creating a naive backend."""
        backend = NaiveAttentionBackend()
        assert backend is not None

    def test_backend_name(self) -> None:
        """Test backend name property."""
        backend = NaiveAttentionBackend()
        assert backend.name == "naive"


class TestFlashInferAvailability:
    """Test FlashInfer availability checking."""

    def test_flashinfer_available_is_bool(self) -> None:
        """Test FLASHINFER_AVAILABLE is boolean."""
        assert isinstance(FLASHINFER_AVAILABLE, bool)
