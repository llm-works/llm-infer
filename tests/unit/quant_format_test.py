"""Unit tests for QuantFormat enum."""

import pytest

from llm_infer.engines.native.backends.linear.formats import QuantFormat

pytestmark = pytest.mark.unit


class TestQuantFormatFromQuantMethod:
    """Test QuantFormat.from_quant_method class method."""

    def test_none_returns_none_format(self) -> None:
        """Test that None method returns NONE format."""
        result = QuantFormat.from_quant_method(None)
        assert result == QuantFormat.NONE

    def test_awq_lowercase(self) -> None:
        """Test AWQ mapping lowercase."""
        result = QuantFormat.from_quant_method("awq")
        assert result == QuantFormat.AWQ

    def test_awq_uppercase(self) -> None:
        """Test AWQ mapping uppercase."""
        result = QuantFormat.from_quant_method("AWQ")
        assert result == QuantFormat.AWQ

    def test_fp8_lowercase(self) -> None:
        """Test FP8 mapping lowercase."""
        result = QuantFormat.from_quant_method("fp8")
        assert result == QuantFormat.FP8

    def test_fp8_mixed_case(self) -> None:
        """Test FP8 mapping mixed case."""
        result = QuantFormat.from_quant_method("Fp8")
        assert result == QuantFormat.FP8

    def test_unknown_returns_none(self) -> None:
        """Test unknown method returns NONE format."""
        result = QuantFormat.from_quant_method("unknown_method")
        assert result == QuantFormat.NONE

    def test_gptq_returns_none(self) -> None:
        """Test unsupported GPTQ returns NONE format."""
        result = QuantFormat.from_quant_method("gptq")
        assert result == QuantFormat.NONE


class TestQuantFormatValues:
    """Test QuantFormat enum values."""

    def test_enum_values_unique(self) -> None:
        """Test that enum values are unique."""
        values = [e.value for e in QuantFormat]
        assert len(values) == len(set(values))

    def test_expected_formats_exist(self) -> None:
        """Test that expected formats are defined."""
        assert hasattr(QuantFormat, "NONE")
        assert hasattr(QuantFormat, "AWQ")
        assert hasattr(QuantFormat, "FP8")
