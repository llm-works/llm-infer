# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Loading compatibility spec files from disk."""

import tempfile

import yaml

from llm_infer.compat import _load_spec_file


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
