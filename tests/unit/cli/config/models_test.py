"""Unit tests for cli/config/models.py (legacy compat module)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_infer.cli.config.models import (
    ModelConfig,
    ModelsConfig,
    SelectionConfig,
    ThinkConfig,
    get_selected_model_name,
    load_models_config,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_reexports_available() -> None:
    """All legacy re-exports should be importable."""
    assert ModelsConfig is not None
    assert ModelConfig is not None
    assert SelectionConfig is not None
    assert ThinkConfig is not None
    assert load_models_config is not None


# ---------------------------------------------------------------------------
# get_selected_model_name (legacy)
# ---------------------------------------------------------------------------


def test_get_selected_model_name_explicit_path(tmp_path: Path) -> None:
    sel = tmp_path / "selected.yaml"
    sel.write_text("name: my-model\n")
    assert get_selected_model_name(sel) == "my-model"


def test_get_selected_model_name_string_path(tmp_path: Path) -> None:
    sel = tmp_path / "selected.yaml"
    sel.write_text("name: my-model\n")
    assert get_selected_model_name(str(sel)) == "my-model"


def test_get_selected_model_name_path_does_not_exist(tmp_path: Path) -> None:
    assert get_selected_model_name(tmp_path / "missing.yaml") is None


def test_get_selected_model_name_no_path_no_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default candidate paths are checked when no path is provided."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert get_selected_model_name() is None


def test_get_selected_model_name_default_candidate_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If ~/.models/selected.yaml exists, it's used."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    candidate = tmp_path / ".models" / "selected.yaml"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("name: from-default\n")
    assert get_selected_model_name() == "from-default"


def test_get_selected_model_name_malformed_yaml(tmp_path: Path) -> None:
    sel = tmp_path / "selected.yaml"
    sel.write_text("not: valid: yaml: [")
    assert get_selected_model_name(sel) is None


def test_get_selected_model_name_empty_file(tmp_path: Path) -> None:
    sel = tmp_path / "selected.yaml"
    sel.write_text("")
    assert get_selected_model_name(sel) is None


def test_get_selected_model_name_no_name_field(tmp_path: Path) -> None:
    sel = tmp_path / "selected.yaml"
    sel.write_text("path: /opt/model\n")
    assert get_selected_model_name(sel) is None
