# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for attention backend selection."""

import pytest

from llm_infer.engines.native import attention
from llm_infer.engines.native.attention import (
    FLASHINFER_AVAILABLE,
    NaiveAttentionBackend,
    get_attention_backend,
)
from llm_infer.engines.native.attention.backends.flashinfer import (
    decode_uses_tensor_cores,
)


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

    def test_flashinfer_unavailable_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that requesting flashinfer when unavailable raises RuntimeError."""
        monkeypatch.setattr(attention, "FLASHINFER_AVAILABLE", False)
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


class TestFlashInferDecodeKernelChoice:
    """Tensor-core decode only where the CUDA-core decode cannot run."""

    @pytest.mark.parametrize("group_size", [1, 2, 3, 4, 6, 8])
    def test_cuda_core_for_supported_group_sizes(self, group_size: int) -> None:
        assert not decode_uses_tensor_cores(group_size * 2, 2)

    @pytest.mark.parametrize("group_size", [5, 7, 12, 16])
    def test_tensor_cores_for_other_group_sizes(self, group_size: int) -> None:
        assert decode_uses_tensor_cores(group_size * 2, 2)
