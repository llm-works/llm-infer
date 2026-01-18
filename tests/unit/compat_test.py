"""Unit tests for compatibility spec generation."""

import tempfile

import pytest
import yaml
from appinfra.log import create_lg

from llm_infer.compat import (
    DEFAULT_ARCHITECTURE_TYPES,
    _get_architectures,
    _load_spec_file,
    _validate_spec,
    check_spec_accuracy,
    get_supported_model_types,
    get_version,
    load_template,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def lg():
    """Create a logger for tests."""
    return create_lg("test", "debug")


class TestGetVersion:
    """Test get_version function."""

    def test_returns_string(self) -> None:
        """Test version returns a string."""
        version = get_version()
        assert isinstance(version, str)

    def test_version_format(self) -> None:
        """Test version is either semver or 'unknown'."""
        version = get_version()
        # Either 'unknown' or x.y.z format
        if version != "unknown":
            parts = version.split(".")
            assert len(parts) >= 2
            assert all(p.isdigit() or p.startswith("dev") for p in parts[:2])


class TestGetSupportedModelTypes:
    """Test get_supported_model_types function."""

    def test_returns_list(self, lg) -> None:
        """Test returns a list of strings."""
        types = get_supported_model_types(lg)
        assert isinstance(types, list)
        assert all(isinstance(t, str) for t in types)

    def test_includes_default_types(self, lg) -> None:
        """Test includes default architecture types."""
        types = get_supported_model_types(lg)
        for default_type in DEFAULT_ARCHITECTURE_TYPES:
            assert default_type in types

    def test_returns_sorted(self, lg) -> None:
        """Test returns sorted list."""
        types = get_supported_model_types(lg)
        assert types == sorted(types)

    def test_includes_registered_architectures(self, lg) -> None:
        """Test includes architectures from ARCHITECTURES registry."""
        types = get_supported_model_types(lg)
        # These are registered in architecture.py
        expected = ["granite", "mistral"]
        for arch in expected:
            assert arch in types


class TestDefaultArchitectureTypes:
    """Test DEFAULT_ARCHITECTURE_TYPES constant."""

    def test_contains_llama(self) -> None:
        """Test llama is in defaults."""
        assert "llama" in DEFAULT_ARCHITECTURE_TYPES

    def test_contains_qwen(self) -> None:
        """Test qwen variants are in defaults."""
        assert "qwen" in DEFAULT_ARCHITECTURE_TYPES
        assert "qwen2" in DEFAULT_ARCHITECTURE_TYPES


class TestGetArchitectures:
    """Test _get_architectures function."""

    def test_returns_dict(self, lg) -> None:
        """Test returns dictionary of architectures."""
        archs = _get_architectures(lg)
        assert isinstance(archs, dict)

    def test_caches_result(self, lg) -> None:
        """Test result is cached on second call."""
        # First call
        archs1 = _get_architectures(lg)
        # Second call should return same object (cached)
        archs2 = _get_architectures(lg)
        assert archs1 is archs2


class TestLoadSpecFile:
    """Test _load_spec_file function."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file returns error."""
        spec, error = _load_spec_file("/nonexistent/path.yaml")
        assert spec is None
        assert "not found" in error.lower()

    def test_load_valid_yaml(self) -> None:
        """Test loading valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"test": "value"}, f)
            f.flush()
            spec, error = _load_spec_file(f.name)
            assert error is None
            assert spec == {"test": "value"}

    def test_load_invalid_yaml(self) -> None:
        """Test loading invalid YAML returns error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            spec, error = _load_spec_file(f.name)
            assert spec is None
            assert "yaml" in error.lower()


class TestValidateSpec:
    """Test _validate_spec function."""

    def test_missing_engine_name(self, lg) -> None:
        """Test detects missing engine name."""
        spec = {"engine": {"name": "wrong"}, "architecture": {}, "version": "1"}
        issues = _validate_spec(lg, spec)
        assert any("engine name" in i.lower() for i in issues)

    def test_missing_sections(self, lg) -> None:
        """Test detects missing required sections."""
        spec = {}
        issues = _validate_spec(lg, spec)
        assert any("missing section" in i.lower() for i in issues)

    def test_valid_spec_minimal_issues(self, lg) -> None:
        """Test valid spec has minimal issues."""
        model_types = get_supported_model_types(lg)
        spec = {
            "version": "1",
            "engine": {"name": "infer"},
            "architecture": {"model_types": model_types},
            "format": {},
            "unsupported_tags": [],
        }
        issues = _validate_spec(lg, spec)
        assert len(issues) == 0


class TestCheckSpecAccuracy:
    """Test check_spec_accuracy function."""

    def test_returns_tuple(self, lg) -> None:
        """Test returns (bool, list) tuple."""
        valid, issues = check_spec_accuracy(lg)
        assert isinstance(valid, bool)
        assert isinstance(issues, list)

    def test_with_nonexistent_file(self, lg) -> None:
        """Test with nonexistent spec file."""
        valid, issues = check_spec_accuracy(lg, spec_file="/nonexistent/file.yaml")
        assert valid is False
        assert len(issues) > 0


class TestLoadTemplate:
    """Test load_template function."""

    def test_loads_template(self) -> None:
        """Test loading the compat template."""
        try:
            template = load_template()
            assert isinstance(template, dict)
            # Should have some expected keys
            assert "engine" in template or "version" in template
        except FileNotFoundError:
            pytest.skip("Template file not found")


class TestValidateSpecExtraTypes:
    """Test _validate_spec with extra/missing model types."""

    def test_extra_model_types(self, lg) -> None:
        """Test detects extra model types not supported."""
        spec = {
            "version": "1",
            "engine": {"name": "infer"},
            "architecture": {"model_types": ["llama", "unsupported_fake_type"]},
            "format": {},
            "unsupported_tags": [],
        }
        issues = _validate_spec(lg, spec)
        assert any("extra model_types" in i.lower() for i in issues)

    def test_missing_model_types(self, lg) -> None:
        """Test detects missing model types."""
        spec = {
            "version": "1",
            "engine": {"name": "infer"},
            "architecture": {"model_types": []},  # Missing all types
            "format": {},
            "unsupported_tags": [],
        }
        issues = _validate_spec(lg, spec)
        assert any("missing model_types" in i.lower() for i in issues)
