# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for model metadata extraction."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest

from llm_infer.models import ModelMetadata, get_model_metadata


class TestModelMetadataProperties:
    """Test ModelMetadata dataclass properties."""

    def test_is_quantized_true_for_quantized(self) -> None:
        """Test is_quantized returns True when quantization is set."""
        meta = ModelMetadata(quantization="bitsandbytes", quantization_bits=4)
        assert meta.is_quantized is True

    def test_is_quantized_false_for_unquantized(self) -> None:
        """Test is_quantized returns False when quantization is None."""
        meta = ModelMetadata(quantization=None)
        assert meta.is_quantized is False

    def test_recommended_fp16_false_for_quantized(self) -> None:
        """Test recommended_fp16 returns False for quantized models."""
        meta = ModelMetadata(quantization="bitsandbytes", quantization_bits=4)
        assert meta.recommended_fp16 is False

    def test_recommended_fp16_true_for_unquantized(self) -> None:
        """Test recommended_fp16 returns True for unquantized models."""
        meta = ModelMetadata(quantization=None)
        assert meta.recommended_fp16 is True

    def test_recommended_bf16_false_for_quantized(self) -> None:
        """Test recommended_bf16 returns False for quantized models."""
        meta = ModelMetadata(quantization="gptq", quantization_bits=4)
        assert meta.recommended_bf16 is False

    def test_recommended_bf16_true_for_unquantized(self) -> None:
        """Test recommended_bf16 returns True for unquantized models."""
        meta = ModelMetadata(quantization=None, torch_dtype="bfloat16")
        assert meta.recommended_bf16 is True


class TestGetModelMetadataErrors:
    """Test get_model_metadata error cases."""

    def test_no_path_or_name_raises(self) -> None:
        """Test that neither path nor name raises ValueError."""
        with pytest.raises(ValueError, match="Either path or name must be provided"):
            get_model_metadata()

    def test_name_without_resolver_raises(self) -> None:
        """Test that name without resolver raises ValueError."""
        with pytest.raises(ValueError, match="resolver required when using name"):
            get_model_metadata(name="some-model")

    @pytest.mark.integration
    def test_missing_config_json_raises(self) -> None:
        """Test that missing config.json raises FileNotFoundError."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(FileNotFoundError),
        ):
            get_model_metadata(path=tmpdir)

    def test_model_not_found_raises(self) -> None:
        """Test that model not found via resolver raises FileNotFoundError."""
        resolver = MagicMock()
        resolver.find_by_name.return_value = None
        with pytest.raises(FileNotFoundError, match="Model not found: missing-model"):
            get_model_metadata(name="missing-model", resolver=resolver)
