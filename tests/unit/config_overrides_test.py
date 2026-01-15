"""Unit tests for config_overrides module."""

from dataclasses import dataclass

import pytest

from llm_infer.serving.dispatch.config_overrides import (
    CliConfigOverride,
    CliOverrides,
    _set_nested_attr,
    parse_override_args,
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


class TestConvertValueInferred:
    """Tests for CliConfigOverride._convert_value_inferred type inference.

    This tests the fallback behavior when target field type is unknown.
    """

    @pytest.fixture
    def converter(self) -> CliConfigOverride:
        """Create a CliConfigOverride instance for testing."""
        return CliConfigOverride(CliOverrides())

    # Boolean handling
    def test_true_lowercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("true") is True

    def test_true_uppercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("TRUE") is True

    def test_yes(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("yes") is True

    def test_false_lowercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("false") is False

    def test_false_uppercase(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("FALSE") is False

    def test_no(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("no") is False

    # Integer handling
    def test_positive_integer(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value_inferred("42")
        assert result == 42
        assert isinstance(result, int)

    def test_negative_integer(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value_inferred("-5")
        assert result == -5
        assert isinstance(result, int)

    def test_zero(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value_inferred("0")
        assert result == 0
        assert isinstance(result, int)

    def test_leading_zeros(self, converter: CliConfigOverride) -> None:
        """Leading zeros should parse as integer (octal not supported)."""
        result = converter._convert_value_inferred("007")
        assert result == 7
        assert isinstance(result, int)

    # Float handling
    def test_positive_float(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value_inferred("3.14")
        assert result == 3.14
        assert isinstance(result, float)

    def test_negative_float(self, converter: CliConfigOverride) -> None:
        result = converter._convert_value_inferred("-2.5")
        assert result == -2.5
        assert isinstance(result, float)

    def test_scientific_notation_whole_becomes_int(
        self, converter: CliConfigOverride
    ) -> None:
        """Scientific notation resulting in whole number becomes int."""
        result = converter._convert_value_inferred("1e3")
        assert result == 1000
        assert isinstance(result, int)

    def test_scientific_notation_large_becomes_int(
        self, converter: CliConfigOverride
    ) -> None:
        """Large scientific notation whole number becomes int."""
        result = converter._convert_value_inferred("1e10")
        assert result == 10000000000
        assert isinstance(result, int)

    def test_scientific_notation_150_becomes_int(
        self, converter: CliConfigOverride
    ) -> None:
        """1.5e2 = 150.0 which is a whole number, so becomes int."""
        result = converter._convert_value_inferred("1.5e2")
        assert result == 150
        assert isinstance(result, int)

    def test_scientific_notation_123_becomes_int(
        self, converter: CliConfigOverride
    ) -> None:
        """1.23e2 = 123.0 which is a whole number, so becomes int."""
        result = converter._convert_value_inferred("1.23e2")
        assert result == 123
        assert isinstance(result, int)

    def test_actual_fractional_result(self, converter: CliConfigOverride) -> None:
        """Scientific notation with actual fractional result stays float."""
        result = converter._convert_value_inferred("1e-1")
        assert result == 0.1
        assert isinstance(result, float)

    # String fallback
    def test_plain_string(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("hello") == "hello"

    def test_string_with_spaces(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("hello world") == "hello world"

    def test_empty_string(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("") == ""

    def test_whitespace_string(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("  ") == "  "

    def test_path_like_string(self, converter: CliConfigOverride) -> None:
        assert converter._convert_value_inferred("/path/to/file") == "/path/to/file"


class TestConvertToType:
    """Tests for CliConfigOverride._convert_to_type with explicit types."""

    @pytest.fixture
    def converter(self) -> CliConfigOverride:
        """Create a CliConfigOverride instance for testing."""
        return CliConfigOverride(CliOverrides())

    def test_convert_to_float(self, converter: CliConfigOverride) -> None:
        """String converted to float stays float, even if whole number."""
        result = converter._convert_to_type("150", float, "test.path")
        assert result == 150.0
        assert isinstance(result, float)

    def test_convert_to_int(self, converter: CliConfigOverride) -> None:
        """String converted to int."""
        result = converter._convert_to_type("42", int, "test.path")
        assert result == 42
        assert isinstance(result, int)

    def test_convert_scientific_to_int(self, converter: CliConfigOverride) -> None:
        """Scientific notation can be converted to int if it's a whole number."""
        result = converter._convert_to_type("1e3", int, "test.path")
        assert result == 1000
        assert isinstance(result, int)

    def test_convert_fractional_to_int_raises(
        self, converter: CliConfigOverride
    ) -> None:
        """Fractional value cannot be converted to int."""
        with pytest.raises(ValueError, match="expected integer"):
            converter._convert_to_type("3.14", int, "test.path")

    def test_convert_to_bool_true(self, converter: CliConfigOverride) -> None:
        """String converted to bool (true variants)."""
        assert converter._convert_to_type("true", bool, "test") is True
        assert converter._convert_to_type("TRUE", bool, "test") is True
        assert converter._convert_to_type("yes", bool, "test") is True
        assert converter._convert_to_type("1", bool, "test") is True

    def test_convert_to_bool_false(self, converter: CliConfigOverride) -> None:
        """String converted to bool (false variants)."""
        assert converter._convert_to_type("false", bool, "test") is False
        assert converter._convert_to_type("FALSE", bool, "test") is False
        assert converter._convert_to_type("no", bool, "test") is False
        assert converter._convert_to_type("0", bool, "test") is False

    def test_convert_invalid_bool_raises(self, converter: CliConfigOverride) -> None:
        """Invalid boolean string raises ValueError."""
        with pytest.raises(ValueError, match="expected boolean"):
            converter._convert_to_type("maybe", bool, "test.path")

    def test_convert_to_str(self, converter: CliConfigOverride) -> None:
        """Any value can be kept as string."""
        assert converter._convert_to_type("hello", str, "test") == "hello"
        assert converter._convert_to_type("123", str, "test") == "123"

    def test_type_mismatch_gives_clear_error(
        self, converter: CliConfigOverride
    ) -> None:
        """Type conversion failure includes path in error."""
        with pytest.raises(ValueError, match="my.config.path") as exc:
            converter._convert_to_type("not-a-number", int, "my.config.path")
        assert "cannot be converted to int" in str(exc.value)


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


class TestTypeValidation:
    """Tests for type-aware validation in generic overrides."""

    def test_float_field_gets_float_value(self) -> None:
        """Float field receives float value, not int."""
        config = MockConfig()
        overrides = CliOverrides(generic={"engines.gpu_memory_utilization": "0.5"})
        CliConfigOverride(overrides).apply(config)
        assert config.engines.gpu_memory_utilization == 0.5
        assert isinstance(config.engines.gpu_memory_utilization, float)

    def test_int_field_gets_int_value(self) -> None:
        """Int field receives int value."""
        config = MockConfig()
        overrides = CliOverrides(generic={"engines.max_batch_size": "64"})
        CliConfigOverride(overrides).apply(config)
        assert config.engines.max_batch_size == 64
        assert isinstance(config.engines.max_batch_size, int)

    def test_bool_field_rejects_invalid_string(self) -> None:
        """Boolean field rejects non-boolean string values."""
        config = MockConfig()
        overrides = CliOverrides(generic={"engines.enforce_eager": "maybe"})
        with pytest.raises(ValueError, match="expected boolean"):
            CliConfigOverride(overrides).apply(config)

    def test_int_field_rejects_fractional(self) -> None:
        """Int field rejects fractional values."""
        config = MockConfig()
        overrides = CliOverrides(generic={"engines.max_batch_size": "3.14"})
        with pytest.raises(ValueError, match="expected integer"):
            CliConfigOverride(overrides).apply(config)

    def test_float_field_rejects_boolean_string(self) -> None:
        """Float field rejects boolean-like strings."""
        config = MockConfig()
        overrides = CliOverrides(generic={"engines.gpu_memory_utilization": "true"})
        with pytest.raises(ValueError, match="cannot be converted to float"):
            CliConfigOverride(overrides).apply(config)

    def test_str_field_accepts_numeric_string(self) -> None:
        """String field keeps numeric values as strings."""
        config = MockConfig()
        overrides = CliOverrides(generic={"api.host": "123"})
        CliConfigOverride(overrides).apply(config)
        assert config.api.host == "123"
        assert isinstance(config.api.host, str)


class TestParseOverrideArgs:
    """Tests for parse_override_args CLI argument parsing."""

    def test_none_input(self) -> None:
        """Test that None input returns None."""
        assert parse_override_args(None) is None

    def test_empty_list(self) -> None:
        """Test that empty list returns None."""
        assert parse_override_args([]) is None

    def test_single_override(self) -> None:
        """Test parsing a single override."""
        result = parse_override_args(["key=value"])
        assert result == {"key": "value"}

    def test_multiple_overrides(self) -> None:
        """Test parsing multiple overrides."""
        result = parse_override_args(["key1=value1", "key2=value2"])
        assert result == {"key1": "value1", "key2": "value2"}

    def test_dotted_key(self) -> None:
        """Test parsing dotted path keys."""
        result = parse_override_args(["engines.vllm.gpu_memory_utilization=0.5"])
        assert result == {"engines.vllm.gpu_memory_utilization": "0.5"}

    def test_value_with_equals(self) -> None:
        """Test that value can contain equals sign."""
        result = parse_override_args(["key=value=with=equals"])
        assert result == {"key": "value=with=equals"}

    def test_empty_value(self) -> None:
        """Test that empty value is allowed."""
        result = parse_override_args(["key="])
        assert result == {"key": ""}

    def test_missing_equals_raises(self) -> None:
        """Test that missing equals sign raises ValueError."""
        with pytest.raises(ValueError, match="expected KEY=VALUE"):
            parse_override_args(["invalid"])

    def test_empty_key_raises(self) -> None:
        """Test that empty key raises ValueError."""
        with pytest.raises(ValueError, match="key cannot be empty"):
            parse_override_args(["=value"])

    def test_whitespace_key_raises(self) -> None:
        """Test that whitespace-only key raises ValueError."""
        with pytest.raises(ValueError, match="key cannot be empty"):
            parse_override_args(["  =value"])

    def test_key_with_leading_whitespace_stripped(self) -> None:
        """Test that leading whitespace in key is stripped."""
        result = parse_override_args(["  key=value"])
        assert result == {"key": "value"}

    def test_key_with_trailing_whitespace_stripped(self) -> None:
        """Test that trailing whitespace in key is stripped."""
        result = parse_override_args(["key  =value"])
        assert result == {"key": "value"}
