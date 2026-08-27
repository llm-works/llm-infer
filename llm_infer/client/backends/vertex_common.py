# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Shared Vertex AI constants and helpers.

Used by both GeminiBackend (OpenAI-compat surface) and NativeVertexBackend
(native REST for cachedContents).
"""

from __future__ import annotations

VALID_SERVICE_TIERS = frozenset({"standard", "priority"})
VERTEX_HOST = "aiplatform.googleapis.com"
VERTEX_PRIORITY_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"
SERVED_TIER_HEADER = "x-gemini-service-tier"


def validate_service_tier(value: str | None) -> str | None:
    """Validate service_tier value. Returns the value if valid, raises ValueError otherwise."""
    if value is None:
        return None
    if value not in VALID_SERVICE_TIERS:
        raise ValueError(
            f"Invalid service_tier {value!r}; "
            f"expected one of {sorted(VALID_SERVICE_TIERS)} or omitted"
        )
    return value
