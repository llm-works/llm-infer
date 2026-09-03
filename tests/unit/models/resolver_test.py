# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for ModelResolver."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.models.config import ModelsConfig
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
