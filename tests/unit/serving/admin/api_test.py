# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""The admin config section; the admin API's handlers are tested in integration."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from llm_infer.serving.dispatch.config import InferenceConfig


def test_admin_config_defaults_disabled_on_localhost() -> None:
    cfg = InferenceConfig.from_dict({}).admin

    assert cfg.enabled is False
    assert cfg.host == "127.0.0.1"
    assert cfg.tokens == []


def test_admin_config_from_dict() -> None:
    raw = {"enabled": True, "port": 9001, "tokens": ["a", "b"], "retry_after": 5}

    cfg = InferenceConfig.from_dict({"admin": raw}).admin

    assert (cfg.enabled, cfg.port, cfg.tokens, cfg.retry_after) == (
        True,
        9001,
        ["a", "b"],
        5,
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"tokenz": ["a"]},  # unknown key
        {"tokens": [""]},  # empty token
        {"tokens": "a"},  # not a list
        {"port": 0},
        {"retry_after": -1},
    ],
)
def test_admin_config_rejects_bad_section(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.from_dict({"admin": raw})
