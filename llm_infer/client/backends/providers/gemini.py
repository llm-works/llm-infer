"""Gemini backend implementation.

This backend extends OpenAICompatibleBackend with Gemini-specific normalization:
- Thinking is disabled by default (matching other providers)
- The `think` flag enables thinking via `reasoning_effort`

Gemini 2.5 models have thinking enabled by default, with thinking tokens counting
against max_output_tokens. This causes issues like structured output truncation.
We normalize this to match other providers where thinking is opt-in.
"""

from __future__ import annotations

from typing import Any

from ...types import ChatRequest
from .openai import OpenAICompatibleBackend


class GeminiBackend(OpenAICompatibleBackend):
    """Backend for Google Gemini via OpenAI-compatible API.

    Normalizes Gemini behavior to match other providers:
    - Thinking disabled by default (reasoning_effort: "none")
    - think=True enables thinking (reasoning_effort: "medium")
    - Explicit reasoning_effort overrides both
    """

    def _build_payload(
        self, request: ChatRequest, messages: list[dict[str, Any]], stream: bool
    ) -> dict[str, Any]:
        """Build payload with Gemini-specific normalization."""
        payload = super()._build_payload(request, messages, stream)
        self._normalize_thinking(payload, request)
        return payload

    def _normalize_thinking(
        self, payload: dict[str, Any], request: ChatRequest
    ) -> None:
        """Normalize thinking behavior to match other providers.

        Gemini 2.5 has thinking enabled by default. We disable it unless
        explicitly requested via think=True or reasoning_effort.

        Also removes the `think` field since Gemini uses `reasoning_effort` instead.

        AI Studio accepts ``reasoning_effort: "none"`` to fully disable thinking.
        Vertex AI's OpenAI-compat surface only accepts ``{high, low, medium,
        minimal}`` and rejects ``"none"`` with HTTP 400; we map to ``"minimal"``
        there — the smallest available budget, not strictly zero.
        """
        # Remove think field - Gemini uses reasoning_effort instead
        payload.pop("think", None)

        # Don't override if user explicitly set reasoning_effort
        if "reasoning_effort" in payload:
            return

        if request.think:
            payload["reasoning_effort"] = "medium"
        else:
            payload["reasoning_effort"] = self._disabled_reasoning_effort()

    def _disabled_reasoning_effort(self) -> str:
        """Value to use for ``reasoning_effort`` when thinking is disabled."""
        if "aiplatform.googleapis.com" in self._base_url:
            return "minimal"
        return "none"
