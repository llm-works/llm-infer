"""Unit tests for Gemini backend thinking normalization."""

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import ChatRequest
from llm_infer.client.backends.providers.gemini import GeminiBackend

pytestmark = pytest.mark.unit


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
