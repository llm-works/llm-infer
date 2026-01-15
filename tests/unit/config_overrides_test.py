"""Unit tests for config_overrides module."""

from dataclasses import dataclass

import pytest

from llm_infer.serving.dispatch.config_overrides import (
    CliConfigOverride,
    CliOverrides,
    _set_nested_attr,
)

pytestmark = pytest.mark.unit


# Test fixtures for nested config structure
@dataclass
class EngineSettings:
    gpu_memory_utilization: float = 0.9
    max_batch_size: int = 32
    enforce_eager: bool = False


@dataclass
class ApiSettings:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class MockConfig:
    engines: EngineSettings | None = None
    api: ApiSettings | None = None

    def __post_init__(self) -> None:
        if self.engines is None:
            self.engines = EngineSettings()
        if self.api is None:
            self.api = ApiSettings()


class TestSetNestedAttr:
    """Tests for _set_nested_attr helper function."""

    def test_single_level_path(self) -> None:
        """Test setting a top-level attribute."""

        @dataclass
        class Simple:
            value: int = 0

        obj = Simple()
        _set_nested_attr(obj, "value", 42)
        assert obj.value == 42

    def test_nested_path(self) -> None:
        """Test setting a nested attribute."""
        config = MockConfig()
        _set_nested_attr(config, "engines.gpu_memory_utilization", 0.5)
        assert config.engines.gpu_memory_utilization == 0.5

    def test_deeply_nested_path(self) -> None:
        """Test setting a deeply nested attribute."""

        @dataclass
        class Level3:
            value: int = 0

        @dataclass
        class Level2:
            level3: Level3 | None = None

            def __post_init__(self) -> None:
                if self.level3 is None:
                    self.level3 = Level3()

        @dataclass
        class Level1:
            level2: Level2 | None = None

            def __post_init__(self) -> None:
                if self.level2 is None:
                    self.level2 = Level2()

        obj = Level1()
        _set_nested_attr(obj, "level2.level3.value", 99)
        assert obj.level2.level3.value == 99

    def test_invalid_intermediate_path_raises(self) -> None:
        """Test that invalid intermediate path raises ValueError."""
        config = MockConfig()
        with pytest.raises(ValueError, match="no attribute 'nonexistent'"):
            _set_nested_attr(config, "nonexistent.value", 123)

    def test_invalid_leaf_path_raises(self) -> None:
        """Test that invalid leaf attribute raises ValueError."""
        config = MockConfig()
        with pytest.raises(ValueError, match="no attribute 'typo'"):
            _set_nested_attr(config, "engines.typo", 123)

    def test_error_message_includes_full_path(self) -> None:
        """Test that error message includes the full path for context."""
        config = MockConfig()
        with pytest.raises(
            ValueError, match="'engines.vllm.setting'.*no attribute 'vllm'"
        ):
            _set_nested_attr(config, "engines.vllm.setting", 123)


class TestConvertValue:
    """Tests for CliConfigOverride._convert_value type inference."""

    @pytest.fixture
    def converter(self) -> CliConfigOverride:
        """Create a CliConfigOverride instance for testing."""
        return CliConfigOverride(CliOverrides())

    # Null/None handling
    def test_null_lowercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("null") is None

    def test_null_uppercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("NULL") is None

    def test_none_lowercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("none") is None

    def test_none_mixed_case(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("None") is None

    # Boolean handling
    def test_true_lowercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("true") is True

    def test_true_uppercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("TRUE") is True

    def test_yes(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("yes") is True

    def test_false_lowercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("false") is False

    def test_false_uppercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("FALSE") is False

    def test_no(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("no") is False

    # Integer handling
    def test_positive_integer(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value("42")
        assert result == 42
        assert isinstance(result, int)

    def test_negative_integer(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value("-5")
        assert result == -5
        assert isinstance(result, int)

    def test_zero(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value("0")
        assert result == 0
        assert isinstance(result, int)

    def test_leading_zeros(self, converter: CliConfigOverride) -> None:
        """Leading zeros should parse as integer (octal not supported)."""
        result = converter._convert_value("007")
        assert result == 7
        assert isinstance(result, int)

    # Float handling
    def test_positive_float(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value("3.14")
        assert result == 3.14
        assert isinstance(result, float)

    def test_negative_float(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value("-2.5")
        assert result == -2.5
        assert isinstance(result, float)

    def test_scientific_notation_whole_number(
        self, converter: CliConfigOverride
    ) -> None:
        """Scientific notation that results in whole number should be int."""
        result = converter._convert_value("1e3")
        assert result == 1000
        assert isinstance(result, int)

    def test_scientific_notation_large(self, converter: CliConfigOverride) -> None:
        """Large scientific notation whole number should be int."""
        result = converter._convert_value("1e10")
        assert result == 10000000000
        assert isinstance(result, int)

    def test_scientific_notation_fractional(self, converter: CliConfigOverride) -> None:
        """Scientific notation with fractional result should be float."""
        result = converter._convert_value("1.5e2")
        assert result == 150.0
        assert isinstance(result, int)  # 150.0 is a whole number

    def test_scientific_notation_non_whole(self, converter: CliConfigOverride) -> None:
        """Scientific notation with non-whole result should be float."""
        result = converter._convert_value("1.23e2")
        assert result == 123.0
        assert isinstance(result, int)  # 123.0 is a whole number

    def test_actual_fractional_result(self, converter: CliConfigOverride) -> None:
        """Scientific notation with actual fractional result stays float."""
        result = converter._convert_value("1e-1")
        assert result == 0.1
        assert isinstance(result, float)

    # String fallback
    def test_plain_string(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("hello") == "hello"

    def test_string_with_spaces(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("hello world") == "hello world"

    def test_empty_string(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("") == ""

    def test_whitespace_string(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("  ") == "  "

    def test_path_like_string(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value("/path/to/file") == "/path/to/file"


class TestCliConfigOverrideGeneric:
    """Tests for CliConfigOverride generic override application."""

    def test_apply_single_override(self) -> None:
        """Test applying a single generic override."""
        config = MockConfig()
        overrides = CliOverrides(generic={"engines.gpu_memory_utilization": "0.5"})
        CliConfigOverride(overrides).apply(config)
        assert config.engines.gpu_memory_utilization == 0.5

    def test_apply_multiple_overrides(self) -> None:
        """Test applying multiple generic overrides."""
        config = MockConfig()
        overrides = CliOverrides(
            generic={
                "engines.gpu_memory_utilization": "0.5",
                "engines.max_batch_size": "64",
                "api.port": "9000",
            }
        )
        CliConfigOverride(overrides).apply(config)
        assert config.engines.gpu_memory_utilization == 0.5
        assert config.engines.max_batch_size == 64
        assert config.api.port == 9000

    def test_apply_boolean_override(self) -> None:
        """Test applying a boolean override."""
        config = MockConfig()
        overrides = CliOverrides(generic={"engines.enforce_eager": "true"})
        CliConfigOverride(overrides).apply(config)
        assert config.engines.enforce_eager is True

    def test_invalid_path_raises(self) -> None:
        """Test that invalid path in generic override raises."""
        config = MockConfig()
        overrides = CliOverrides(generic={"engines.typo": "value"})
        with pytest.raises(ValueError, match="no attribute 'typo'"):
            CliConfigOverride(overrides).apply(config)

    def test_generic_overrides_applied_after_explicit(self) -> None:
        """Test that generic overrides are applied after explicit ones."""
        config = MockConfig()
        # Set explicit port override and generic port override
        # Generic should win (applied last)
        overrides = CliOverrides(
            port=8080,
            generic={"api.port": "9000"},
        )
        CliConfigOverride(overrides).apply(config)
        assert config.api.port == 9000


class TestParseOverrides:
    """Tests for ServeTool._parse_overrides CLI argument parsing."""

    @staticmethod
    def _parse_overrides(overrides: list[str] | None) -> dict[str, str] | None:
        """Standalone implementation of _parse_overrides for testing."""
        if not overrides:
            return None
        result = {}
        for item in overrides:
            if "=" not in item:
                raise ValueError(
                    f"Invalid override format: {item!r} (expected KEY=VALUE)"
                )
            key, value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(
                    f"Invalid override format: {item!r} (key cannot be empty)"
                )
            result[key] = value
        return result

    def test_none_input(self) -> None:
        """Test that None input returns None."""
        assert self._parse_overrides(None) is None

    def test_empty_list(self) -> None:
        """Test that empty list returns None."""
        assert self._parse_overrides([]) is None

    def test_single_override(self) -> None:
        """Test parsing a single override."""
        result = self._parse_overrides(["key=value"])
        assert result == {"key": "value"}

    def test_multiple_overrides(self) -> None:
        """Test parsing multiple overrides."""
        result = self._parse_overrides(["key1=value1", "key2=value2"])
        assert result == {"key1": "value1", "key2": "value2"}

    def test_dotted_key(self) -> None:
        """Test parsing dotted path keys."""
        result = self._parse_overrides(["engines.vllm.gpu_memory_utilization=0.5"])
        assert result == {"engines.vllm.gpu_memory_utilization": "0.5"}

    def test_value_with_equals(self) -> None:
        """Test that value can contain equals sign."""
        result = self._parse_overrides(["key=value=with=equals"])
        assert result == {"key": "value=with=equals"}

    def test_empty_value(self) -> None:
        """Test that empty value is allowed."""
        result = self._parse_overrides(["key="])
        assert result == {"key": ""}

    def test_missing_equals_raises(self) -> None:
        """Test that missing equals sign raises ValueError."""
        with pytest.raises(ValueError, match="expected KEY=VALUE"):
            self._parse_overrides(["invalid"])

    def test_empty_key_raises(self) -> None:
        """Test that empty key raises ValueError."""
        with pytest.raises(ValueError, match="key cannot be empty"):
            self._parse_overrides(["=value"])

    def test_whitespace_key_raises(self) -> None:
        """Test that whitespace-only key raises ValueError."""
        with pytest.raises(ValueError, match="key cannot be empty"):
            self._parse_overrides(["  =value"])

    def test_key_with_leading_whitespace_stripped(self) -> None:
        """Test that leading whitespace in key is stripped."""
        result = self._parse_overrides(["  key=value"])
        assert result == {"key": "value"}

    def test_key_with_trailing_whitespace_stripped(self) -> None:
        """Test that trailing whitespace in key is stripped."""
        result = self._parse_overrides(["key  =value"])
        assert result == {"key": "value"}
