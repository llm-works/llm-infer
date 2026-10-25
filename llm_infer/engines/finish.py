# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Finish-reason parsing for engine backends."""

from __future__ import annotations

from ..schemas.openai import FinishReason


def parse_finish_reason(raw: str | None) -> FinishReason | None:
    """Map an upstream finish_reason string to FinishReason.

    None stays None (the upstream reported nothing). Values outside the OpenAI
    set, such as vLLM's "abort", map to STOP, which is how the serving layer
    has always reported them to clients.
    """
    if raw is None:
        return None
    try:
        return FinishReason(raw)
    except ValueError:
        return FinishReason.STOP
