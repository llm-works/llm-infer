# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for Gemini backend thinking normalization."""

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import ChatRequest, ChatResponse
from llm_infer.client.backends.providers.gemini import GeminiBackend

pytestmark = pytest.mark.unit

_VERTEX_BASE_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/"
    "p/locations/us-central1/endpoints/openapi"
)
_STUDIO_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_VERTEX_PRIORITY_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger for testing."""
    return MagicMock(spec=Logger)


class TestGeminiBackendThinkingNormalization:
    """Test GeminiBackend thinking normalization."""

    def test_thinking_disabled_by_default(self, mock_lg: Logger) -> None:
        """Test thinking is disabled by default (reasoning_effort: none)."""
        backend = GeminiBackend(mock_lg, "gemini")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(messages=messages, model="gemini-2.5-flash")
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["reasoning_effort"] == "none"
        assert "think" not in payload  # think field should be removed
        backend.close()

    def test_thinking_enabled_with_think_flag(self, mock_lg: Logger) -> None:
        """Test think=True enables thinking (reasoning_effort: medium)."""
        backend = GeminiBackend(mock_lg, "gemini")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(messages=messages, model="gemini-2.5-flash", think=True)
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["reasoning_effort"] == "medium"
        assert "think" not in payload  # think field should be removed
        backend.close()

    def test_explicit_reasoning_effort_not_overridden(self, mock_lg: Logger) -> None:
        """Test explicit reasoning_effort in extra is preserved."""
        backend = GeminiBackend(mock_lg, "gemini")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            messages=messages,
            model="gemini-2.5-flash",
            extra={"reasoning_effort": "high"},
        )
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["reasoning_effort"] == "high"
        backend.close()

    def test_explicit_reasoning_effort_overrides_think_flag(
        self, mock_lg: Logger
    ) -> None:
        """Test explicit reasoning_effort takes precedence over think flag."""
        backend = GeminiBackend(mock_lg, "gemini")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            messages=messages,
            model="gemini-2.5-flash",
            think=True,
            extra={"reasoning_effort": "low"},
        )
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["reasoning_effort"] == "low"
        backend.close()

    def test_streaming_also_normalized(self, mock_lg: Logger) -> None:
        """Test streaming requests also get thinking normalization."""
        backend = GeminiBackend(mock_lg, "gemini")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(messages=messages, model="gemini-2.5-flash")
        payload = backend._build_payload(request, messages, stream=True)

        assert payload["reasoning_effort"] == "none"
        assert payload["stream"] is True
        backend.close()

    def test_structured_output_works_with_disabled_thinking(
        self, mock_lg: Logger
    ) -> None:
        """Test structured output with disabled thinking (the original issue)."""
        backend = GeminiBackend(mock_lg, "gemini")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            messages=messages,
            model="gemini-2.5-flash",
            extra={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "test", "schema": {"type": "object"}},
                }
            },
        )
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["reasoning_effort"] == "none"
        assert payload["response_format"]["type"] == "json_schema"
        backend.close()


class TestGeminiBackendVertexReasoningEffort:
    """Vertex's OpenAI-compat surface rejects reasoning_effort='none'.

    AI Studio accepts ``{high, low, medium, minimal, none}``; Vertex accepts
    only ``{high, low, medium, minimal}``. The backend picks the right value
    based on ``base_url``.
    """

    def test_disabled_thinking_uses_minimal_on_vertex(self, mock_lg: Logger) -> None:
        backend = GeminiBackend(
            mock_lg,
            "vertex",
            base_url=(
                "https://us-central1-aiplatform.googleapis.com/v1/projects/"
                "p/locations/us-central1/endpoints/openapi"
            ),
        )
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(messages=messages, model="google/gemini-2.5-flash")
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["reasoning_effort"] == "minimal"
        backend.close()

    def test_enabled_thinking_still_medium_on_vertex(self, mock_lg: Logger) -> None:
        """think=True still maps to medium regardless of provider surface."""
        backend = GeminiBackend(
            mock_lg,
            "vertex",
            base_url="https://us-central1-aiplatform.googleapis.com/v1/...",
        )
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            messages=messages, model="google/gemini-2.5-flash", think=True
        )
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["reasoning_effort"] == "medium"
        backend.close()

    def test_ai_studio_keeps_none(self, mock_lg: Logger) -> None:
        """AI Studio base_url still produces 'none' (backwards compatible)."""
        backend = GeminiBackend(
            mock_lg,
            "studio",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(messages=messages, model="gemini-2.5-flash")
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["reasoning_effort"] == "none"
        backend.close()


class TestGeminiBackendServiceTier:
    """Vertex AI Priority service tier behavior."""

    def _request(self) -> tuple[ChatRequest, list[dict[str, object]]]:
        messages: list[dict[str, object]] = [{"role": "user", "content": "Hi"}]
        return ChatRequest(messages=messages, model="gemini-2.5-flash"), messages

    def test_omitted_sends_no_header_or_body_param(self, mock_lg: Logger) -> None:
        backend = GeminiBackend(mock_lg, "vertex", base_url=_VERTEX_BASE_URL)
        request, messages = self._request()
        payload = backend._build_payload(request, messages, stream=False)
        headers = backend._build_headers()

        assert "service_tier" not in payload
        assert _VERTEX_PRIORITY_HEADER not in headers
        backend.close()

    def test_standard_sends_no_header_or_body_param(self, mock_lg: Logger) -> None:
        backend = GeminiBackend(
            mock_lg, "vertex", base_url=_VERTEX_BASE_URL, service_tier="standard"
        )
        request, messages = self._request()
        payload = backend._build_payload(request, messages, stream=False)
        headers = backend._build_headers()

        assert "service_tier" not in payload
        assert _VERTEX_PRIORITY_HEADER not in headers
        backend.close()

    def test_priority_on_vertex_injects_header_and_body_param(
        self, mock_lg: Logger
    ) -> None:
        backend = GeminiBackend(
            mock_lg, "vertex", base_url=_VERTEX_BASE_URL, service_tier="priority"
        )
        request, messages = self._request()
        payload = backend._build_payload(request, messages, stream=False)
        headers = backend._build_headers()

        assert payload["service_tier"] == "priority"
        assert headers[_VERTEX_PRIORITY_HEADER] == "priority"
        backend.close()

    def test_priority_on_studio_omits_vertex_header(self, mock_lg: Logger) -> None:
        """Studio uses the body-param spelling; the Vertex header is Vertex-only."""
        backend = GeminiBackend(
            mock_lg, "studio", base_url=_STUDIO_BASE_URL, service_tier="priority"
        )
        request, messages = self._request()
        payload = backend._build_payload(request, messages, stream=False)
        headers = backend._build_headers()

        assert payload["service_tier"] == "priority"
        assert _VERTEX_PRIORITY_HEADER not in headers
        mock_lg.warning.assert_called_once()
        backend.close()

    def test_invalid_service_tier_rejected(self, mock_lg: Logger) -> None:
        with pytest.raises(ValueError, match="Invalid service_tier"):
            GeminiBackend(
                mock_lg, "vertex", base_url=_VERTEX_BASE_URL, service_tier="express"
            )

    def test_priority_explicit_extra_overrides_body_default(
        self, mock_lg: Logger
    ) -> None:
        """If a caller already specifies service_tier in extra, don't clobber."""
        backend = GeminiBackend(
            mock_lg, "vertex", base_url=_VERTEX_BASE_URL, service_tier="priority"
        )
        messages: list[dict[str, object]] = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            messages=messages,
            model="gemini-2.5-flash",
            extra={"service_tier": "auto"},
        )
        payload = backend._build_payload(request, messages, stream=False)

        assert payload["service_tier"] == "auto"
        backend.close()

    @pytest.mark.asyncio
    async def test_priority_header_on_async_path(self, mock_lg: Logger) -> None:
        backend = GeminiBackend(
            mock_lg, "vertex", base_url=_VERTEX_BASE_URL, service_tier="priority"
        )
        headers = await backend._build_headers_async()
        assert headers[_VERTEX_PRIORITY_HEADER] == "priority"
        await backend.aclose()


class TestGeminiBackendDowngradeLogging:
    """Backend emits WARN when priority was requested but served as standard."""

    def _make_response(self, served_tier: str | None) -> ChatResponse:
        return ChatResponse(
            content="hi",
            model="gemini-2.5-flash",
            headers={"x-gemini-service-tier": served_tier} if served_tier else None,
        )

    def test_no_warning_when_served_priority(self, mock_lg: Logger) -> None:
        backend = GeminiBackend(
            mock_lg, "vertex", base_url=_VERTEX_BASE_URL, service_tier="priority"
        )
        mock_lg.warning.reset_mock()
        request, _ = TestGeminiBackendServiceTier()._request()
        backend._after_response(request, self._make_response("priority"))
        mock_lg.warning.assert_not_called()
        backend.close()

    def test_warns_when_served_standard(self, mock_lg: Logger) -> None:
        backend = GeminiBackend(
            mock_lg, "vertex", base_url=_VERTEX_BASE_URL, service_tier="priority"
        )
        mock_lg.warning.reset_mock()
        request, _ = TestGeminiBackendServiceTier()._request()
        backend._after_response(request, self._make_response("standard"))

        mock_lg.warning.assert_called_once()
        _, kwargs = mock_lg.warning.call_args
        assert kwargs["extra"]["tier_requested"] == "priority"
        assert kwargs["extra"]["tier_served"] == "standard"
        backend.close()

    def test_no_warning_when_priority_not_configured(self, mock_lg: Logger) -> None:
        """A backend without service_tier='priority' never logs downgrades."""
        backend = GeminiBackend(mock_lg, "vertex", base_url=_VERTEX_BASE_URL)
        mock_lg.warning.reset_mock()
        request, _ = TestGeminiBackendServiceTier()._request()
        backend._after_response(request, self._make_response("standard"))
        mock_lg.warning.assert_not_called()
        backend.close()

    def test_no_warning_when_header_absent(self, mock_lg: Logger) -> None:
        """Server didn't echo the tier header at all - skip without warning."""
        backend = GeminiBackend(
            mock_lg, "vertex", base_url=_VERTEX_BASE_URL, service_tier="priority"
        )
        mock_lg.warning.reset_mock()
        request, _ = TestGeminiBackendServiceTier()._request()
        backend._after_response(request, self._make_response(None))
        mock_lg.warning.assert_not_called()
        backend.close()

    def test_no_warning_on_studio_backend(self, mock_lg: Logger) -> None:
        """Studio backend should not emit Vertex-specific downgrade warning."""
        backend = GeminiBackend(
            mock_lg, "studio", base_url=_STUDIO_BASE_URL, service_tier="priority"
        )
        mock_lg.warning.reset_mock()
        request, _ = TestGeminiBackendServiceTier()._request()
        backend._after_response(request, self._make_response("standard"))
        mock_lg.warning.assert_not_called()
        backend.close()
