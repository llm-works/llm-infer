"""Unit tests for provider detection."""

import pytest

from llm_infer.client.backends.provider import Provider, ProviderDetector

pytestmark = pytest.mark.unit


class TestProviderDetector:
    """Test ProviderDetector.detect()."""

    def test_detect_anthropic_from_url(self) -> None:
        """Test Anthropic detection from URL."""
        assert (
            ProviderDetector.detect("https://api.anthropic.com/v1")
            == Provider.ANTHROPIC
        )
        assert (
            ProviderDetector.detect("https://api.anthropic.com") == Provider.ANTHROPIC
        )

    def test_detect_openai_from_url(self) -> None:
        """Test OpenAI detection from URL."""
        assert ProviderDetector.detect("https://api.openai.com/v1") == Provider.OPENAI
        assert ProviderDetector.detect("https://api.openai.com") == Provider.OPENAI

    def test_detect_xai_from_url(self) -> None:
        """Test xAI detection from URL."""
        assert ProviderDetector.detect("https://api.x.ai/v1") == Provider.XAI
        assert ProviderDetector.detect("https://api.x.ai") == Provider.XAI

    def test_detect_google_from_url(self) -> None:
        """Test Google detection from URL."""
        assert (
            ProviderDetector.detect("https://generativelanguage.googleapis.com/v1")
            == Provider.GOOGLE
        )
        assert (
            ProviderDetector.detect("https://aiplatform.googleapis.com/v1")
            == Provider.GOOGLE
        )

    def test_detect_azure_from_url(self) -> None:
        """Test Azure detection from URL."""
        assert (
            ProviderDetector.detect("https://myresource.openai.azure.com/v1")
            == Provider.AZURE
        )

    def test_detect_local_from_url(self) -> None:
        """Test local detection from URL."""
        assert ProviderDetector.detect("http://localhost:8000/v1") == Provider.LOCAL
        assert ProviderDetector.detect("http://127.0.0.1:8000/v1") == Provider.LOCAL
        assert ProviderDetector.detect("http://0.0.0.0:8000") == Provider.LOCAL

    def test_detect_anthropic_from_api_key(self) -> None:
        """Test Anthropic detection from API key prefix."""
        assert ProviderDetector.detect(api_key="sk-ant-api03-xxx") == Provider.ANTHROPIC

    def test_detect_unknown_url(self) -> None:
        """Test unknown provider from unrecognized URL."""
        assert (
            ProviderDetector.detect("https://custom-llm.example.com/v1")
            == Provider.UNKNOWN
        )

    def test_detect_unknown_no_args(self) -> None:
        """Test unknown provider when no args provided."""
        assert ProviderDetector.detect() == Provider.UNKNOWN

    def test_url_takes_precedence_over_key(self) -> None:
        """Test that URL pattern takes precedence over API key."""
        # Even with an Anthropic key, URL wins
        assert (
            ProviderDetector.detect("https://api.openai.com/v1", "sk-ant-xxx")
            == Provider.OPENAI
        )


class TestProviderEnum:
    """Test Provider enum."""

    def test_provider_values(self) -> None:
        """Test provider string values."""
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.XAI.value == "xai"
        assert Provider.GOOGLE.value == "google"
        assert Provider.AZURE.value == "azure"
        assert Provider.LOCAL.value == "local"
        assert Provider.UNKNOWN.value == "unknown"

    def test_provider_is_string(self) -> None:
        """Test provider can be used as string."""
        assert Provider.OPENAI == "openai"
        assert Provider.OPENAI.value == "openai"
