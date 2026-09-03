# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for ModelsConfig.from_dict — default-model parsing."""

from __future__ import annotations

import pytest

from llm_infer.models.config import ModelsConfig

pytestmark = pytest.mark.unit


def test_from_dict_default_absent_yields_none() -> None:
    cfg = ModelsConfig.from_dict({"models": {"m": {}}})
    assert cfg.default is None


def test_from_dict_default_string_is_parsed() -> None:
    cfg = ModelsConfig.from_dict({"default": "qwen2.5-0.5b", "models": {}})
    assert cfg.default == "qwen2.5-0.5b"


def test_from_dict_legacy_selection_block_is_ignored() -> None:
    """Stale `selection:` blocks from pre-v1 configs must parse cleanly.

    Protects ops's existing models.yaml files during the transition to
    top-level `default:`.
    """
    cfg = ModelsConfig.from_dict(
        {
            "selection": {
                "generate": {"path": None, "default": "should-be-ignored"},
                "embed": {"path": None, "default": None},
            },
            "models": {},
        }
    )
    assert cfg.default is None
