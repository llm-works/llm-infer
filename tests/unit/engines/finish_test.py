# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for engines/finish.py."""

from __future__ import annotations

import pytest

from llm_infer.engines.finish import parse_finish_reason
from llm_infer.schemas.openai import FinishReason

pytestmark = pytest.mark.unit


class TestParseFinishReason:
    def test_known_values(self) -> None:
        assert parse_finish_reason("length") is FinishReason.LENGTH
        assert parse_finish_reason("tool_calls") is FinishReason.TOOL_CALLS
        assert parse_finish_reason("stop") is FinishReason.STOP

    def test_none_stays_none(self) -> None:
        assert parse_finish_reason(None) is None

    def test_unknown_upstream_value_maps_to_stop(self) -> None:
        """vLLM's "abort" has no OpenAI equivalent; clients have always seen stop."""
        assert parse_finish_reason("abort") is FinishReason.STOP
