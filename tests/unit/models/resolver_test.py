"""Unit tests for ModelResolver."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.models.config import ModelsConfig, SelectionConfig
from llm_infer.models.resolver import ModelResolver, create_resolver

pytestmark = pytest.mark.unit


@pytest.fixture
def lg() -> Logger:
    return MagicMock(spec=Logger)


def _make_model_dir(parent: Path, name: str) -> Path:
    """Create a fake model directory with config.json (marker file)."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    return d


# ---------------------------------------------------------------------------
# find_by_name
# ---------------------------------------------------------------------------


class TestFindByName:
    def test_finds_in_first_location(self, lg: Logger, tmp_path: Path) -> None:
        loc1 = tmp_path / "loc1"
        loc1.mkdir()
        _make_model_dir(loc1, "qwen-7b")
        r = ModelResolver(lg, [loc1])
        result = r.find_by_name("qwen-7b")
        assert result is not None
        assert result.name == "qwen-7b"

    def test_searches_locations_in_order(self, lg: Logger, tmp_path: Path) -> None:
        loc1 = tmp_path / "loc1"
        loc2 = tmp_path / "loc2"
        loc1.mkdir()
        loc2.mkdir()
        _make_model_dir(loc2, "qwen-7b")
        r = ModelResolver(lg, [loc1, loc2])
        result = r.find_by_name("qwen-7b")
        assert result is not None
        assert str(loc2) in str(result)

    def test_not_found_returns_none(self, lg: Logger, tmp_path: Path) -> None:
        loc1 = tmp_path / "loc1"
        loc1.mkdir()
        r = ModelResolver(lg, [loc1])
        assert r.find_by_name("missing") is None

    def test_directory_without_config_json_skipped(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        loc1 = tmp_path / "loc1"
        loc1.mkdir()
        (loc1 / "incomplete-model").mkdir()  # No config.json
        r = ModelResolver(lg, [loc1])
        assert r.find_by_name("incomplete-model") is None


# ---------------------------------------------------------------------------
# load_selection_file
# ---------------------------------------------------------------------------


class TestLoadSelectionFile:
    def test_load_with_name(self, lg: Logger, tmp_path: Path) -> None:
        sel = tmp_path / "sel.yaml"
        sel.write_text("name: qwen-7b\n")
        r = ModelResolver(lg, [])
        name, path = r.load_selection_file(sel)
        assert name == "qwen-7b"
        assert path is None

    def test_load_with_path(self, lg: Logger, tmp_path: Path) -> None:
        sel = tmp_path / "sel.yaml"
        sel.write_text("path: /opt/models/qwen-7b\n")
        r = ModelResolver(lg, [])
        name, path = r.load_selection_file(sel)
        assert name is None
        assert path == Path("/opt/models/qwen-7b")

    def test_load_empty_file(self, lg: Logger, tmp_path: Path) -> None:
        sel = tmp_path / "sel.yaml"
        sel.write_text("")
        r = ModelResolver(lg, [])
        name, path = r.load_selection_file(sel)
        assert name is None
        assert path is None

    def test_missing_file_returns_none(self, lg: Logger, tmp_path: Path) -> None:
        r = ModelResolver(lg, [])
        name, path = r.load_selection_file(tmp_path / "missing.yaml")
        assert name is None
        assert path is None
        # debug logged for missing file
        assert lg.debug.called  # type: ignore[attr-defined]

    def test_malformed_yaml_logs_warning(self, lg: Logger, tmp_path: Path) -> None:
        sel = tmp_path / "sel.yaml"
        sel.write_text("not: valid: yaml: [")
        r = ModelResolver(lg, [])
        name, path = r.load_selection_file(sel)
        assert name is None
        assert path is None
        assert lg.warning.called  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# resolve - direct path
# ---------------------------------------------------------------------------


class TestResolveDirectPath:
    def test_existing_path(self, lg: Logger, tmp_path: Path) -> None:
        d = _make_model_dir(tmp_path, "model")
        r = ModelResolver(lg, [])
        assert r.resolve(model_path=d) == d

    def test_nonexistent_path_logs_error(self, lg: Logger, tmp_path: Path) -> None:
        r = ModelResolver(lg, [])
        assert r.resolve(model_path=tmp_path / "missing") is None
        assert lg.error.called  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# resolve - by name
# ---------------------------------------------------------------------------


class TestResolveByName:
    def test_found_in_locations(self, lg: Logger, tmp_path: Path) -> None:
        loc = tmp_path / "loc"
        loc.mkdir()
        _make_model_dir(loc, "model")
        r = ModelResolver(lg, [loc])
        assert r.resolve(model_name="model") is not None

    def test_not_found_logs_error(self, lg: Logger, tmp_path: Path) -> None:
        loc = tmp_path / "loc"
        loc.mkdir()
        r = ModelResolver(lg, [loc])
        assert r.resolve(model_name="missing") is None
        assert lg.error.called  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# resolve - selection
# ---------------------------------------------------------------------------


class TestResolveSelection:
    def test_selection_file_with_path(self, lg: Logger, tmp_path: Path) -> None:
        d = _make_model_dir(tmp_path, "selected-model")
        sel_file = tmp_path / "sel.yaml"
        sel_file.write_text(f"path: {d}\n")
        r = ModelResolver(lg, [])
        result = r.resolve(selection=SelectionConfig(path=str(sel_file)))
        assert result == d

    def test_selection_file_with_name_resolves_in_locations(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        loc = tmp_path / "loc"
        loc.mkdir()
        _make_model_dir(loc, "selected-model")
        sel_file = tmp_path / "sel.yaml"
        sel_file.write_text("name: selected-model\n")
        r = ModelResolver(lg, [loc])
        result = r.resolve(selection=SelectionConfig(path=str(sel_file)))
        assert result is not None
        assert result.name == "selected-model"

    def test_selection_file_with_name_not_found(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        loc = tmp_path / "loc"
        loc.mkdir()
        sel_file = tmp_path / "sel.yaml"
        sel_file.write_text("name: missing-model\n")
        r = ModelResolver(lg, [loc])
        result = r.resolve(selection=SelectionConfig(path=str(sel_file)))
        assert result is None
        assert lg.error.called  # type: ignore[attr-defined]

    def test_selection_path_does_not_exist(self, lg: Logger, tmp_path: Path) -> None:
        sel_file = tmp_path / "sel.yaml"
        sel_file.write_text(f"path: {tmp_path}/missing-dir\n")
        r = ModelResolver(lg, [])
        result = r.resolve(selection=SelectionConfig(path=str(sel_file)))
        assert result is None
        assert lg.error.called  # type: ignore[attr-defined]

    def test_falls_back_to_default(self, lg: Logger, tmp_path: Path) -> None:
        loc = tmp_path / "loc"
        loc.mkdir()
        _make_model_dir(loc, "default-model")
        # No selection file, only default
        r = ModelResolver(lg, [loc])
        result = r.resolve(
            selection=SelectionConfig(path=None, default="default-model")
        )
        assert result is not None

    def test_default_not_found(self, lg: Logger, tmp_path: Path) -> None:
        loc = tmp_path / "loc"
        loc.mkdir()
        r = ModelResolver(lg, [loc])
        result = r.resolve(
            selection=SelectionConfig(path=None, default="missing-default")
        )
        assert result is None
        assert lg.error.called  # type: ignore[attr-defined]

    def test_selection_file_missing_falls_back_to_default(
        self, lg: Logger, tmp_path: Path
    ) -> None:
        loc = tmp_path / "loc"
        loc.mkdir()
        _make_model_dir(loc, "default-model")
        r = ModelResolver(lg, [loc])
        result = r.resolve(
            selection=SelectionConfig(
                path=str(tmp_path / "missing-sel.yaml"),
                default="default-model",
            )
        )
        assert result is not None

    def test_no_path_no_default_returns_none(self, lg: Logger, tmp_path: Path) -> None:
        r = ModelResolver(lg, [])
        result = r.resolve(selection=SelectionConfig(path=None, default=None))
        assert result is None


# ---------------------------------------------------------------------------
# resolve - no input
# ---------------------------------------------------------------------------


def test_resolve_with_nothing_logs_error(lg: Logger) -> None:
    r = ModelResolver(lg, [])
    assert r.resolve() is None
    assert lg.error.called  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# create_resolver factory
# ---------------------------------------------------------------------------


def test_create_resolver(lg: Logger, tmp_path: Path) -> None:
    config = ModelsConfig(locations=[tmp_path / "loc1", tmp_path / "loc2"])
    r = create_resolver(lg, config)
    assert isinstance(r, ModelResolver)
    assert r.locations == [tmp_path / "loc1", tmp_path / "loc2"]
