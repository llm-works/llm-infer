"""Gemini backend implementation.

This backend extends OpenAICompatibleBackend with Gemini-specific normalization:
- Thinking is disabled by default (matching other providers)
- The `think` flag enables thinking via `reasoning_effort`
- Optional Vertex AI Priority service tier

Gemini 2.5 models have thinking enabled by default, with thinking tokens counting
against max_output_tokens. This causes issues like structured output truncation.
We normalize this to match other providers where thinking is opt-in.
"""

from __future__ import annotations

from typing import Any

from appinfra.log import Logger

from ...types import ChatRequest, ChatResponse
from ..auth import AuthProvider
from ..context import BackendContext
from .openai import OpenAICompatibleBackend

_VALID_SERVICE_TIERS = frozenset({"standard", "priority"})
_VERTEX_HOST = "aiplatform.googleapis.com"
_VERTEX_PRIORITY_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"
_SERVED_TIER_HEADER = "x-gemini-service-tier"


class GeminiBackend(OpenAICompatibleBackend):
    """Backend for Google Gemini via OpenAI-compatible API.

    Normalizes Gemini behavior to match other providers:
    - Thinking disabled by default (reasoning_effort: "none")
    - think=True enables thinking (reasoning_effort: "medium")
    - Explicit reasoning_effort overrides both

    Optional Vertex AI Priority service tier (``service_tier="priority"``)
    adds the ``X-Vertex-AI-LLM-Shared-Request-Type: priority`` header on
    every chat request and a ``service_tier: "priority"`` body param.
    Priority is a Vertex-only feature; on AI Studio the header is skipped
    and only the body param is sent (Studio's own spelling). Priority can
    silently downgrade under capacity events — inspect
    ``ChatResponse.headers["x-gemini-service-tier"]`` to detect this
    downstream (e.g. emit a metric); the backend also emits a WARN log on
    observed downgrades for visibility without callback wiring.
    """

    def __init__(
        self,
        lg: Logger,
        name: str,
        ctx: BackendContext | None = None,
        default_model: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        auth: AuthProvider | None = None,
        service_tier: str | None = None,
    ) -> None:
        super().__init__(lg, name, ctx, default_model, base_url, api_key, auth)
        self._service_tier = self._validate_service_tier(service_tier)
        self._is_vertex = _VERTEX_HOST in self._base_url
        if self._service_tier == "priority" and not self._is_vertex:
            self._lg.warning(
                "service_tier='priority' configured against non-Vertex "
                "endpoint; Vertex Priority header will not be sent",
                extra={"backend": name, "base_url": base_url},
            )

    @staticmethod
    def _validate_service_tier(value: str | None) -> str | None:
        if value is None:
            return None
        if value not in _VALID_SERVICE_TIERS:
            raise ValueError(
                f"Invalid service_tier {value!r}; "
                f"expected one of {sorted(_VALID_SERVICE_TIERS)} or omitted"
            )
        return value

    def _build_headers(self) -> dict[str, str]:
        headers = super()._build_headers()
        if self._service_tier == "priority" and self._is_vertex:
            headers[_VERTEX_PRIORITY_HEADER] = "priority"
        return headers

    async def _build_headers_async(self) -> dict[str, str]:
        headers = await super()._build_headers_async()
        if self._service_tier == "priority" and self._is_vertex:
            headers[_VERTEX_PRIORITY_HEADER] = "priority"
        return headers

    def _build_payload(
        self, request: ChatRequest, messages: list[dict[str, Any]], stream: bool
    ) -> dict[str, Any]:
        """Build payload with Gemini-specific normalization."""
        payload = super()._build_payload(request, messages, stream)
        self._normalize_thinking(payload, request)
        if self._service_tier == "priority":
            payload.setdefault("service_tier", "priority")
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
        if self._is_vertex:
            return "minimal"
        return "none"

    def _after_response(self, request: ChatRequest, response: ChatResponse) -> None:
        """Log a downgrade when priority was requested but not served.

        Vertex Priority can silently fall back to Standard under capacity
        pressure; the served tier is reported in the
        ``x-gemini-service-tier`` response header. Logged at WARN so it
        surfaces in default logging without a callback wired in.
        """
        # As of 2026-06-24, Vertex's OpenAI-compat /chat/completions does
        # not emit `x-gemini-service-tier` for gemini-2.5-flash, so this
        # branch is currently a no-op (kept for when Google adds it).
        if self._service_tier != "priority" or not response.headers:
            return
        served = response.headers.get(_SERVED_TIER_HEADER)
        if served is None or served == "priority":
            return
        self._lg.warning(
            "vertex priority downgraded",
            extra={
                "backend": self._name,
                "model": response.model,
                "tier_requested": "priority",
                "tier_served": served,
            },
        )
