"""Unit tests for BackendFactory."""

from unittest.mock import MagicMock

import pytest
from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from llm_infer.client.backends import BackendFactory
from llm_infer.client.backends.providers.gemini import GeminiBackend
from llm_infer.client.backends.providers.openai import OpenAICompatibleBackend

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger for testing."""
    return MagicMock(spec=Logger)


class TestBackendFactoryProviderDetection:
    """Test BackendFactory creates correct backend based on provider."""

    def test_creates_gemini_backend_for_google_url(self, mock_lg: Logger) -> None:
        """Test factory creates GeminiBackend for Google URLs."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "model": "gemini-2.5-flash",
            }
        )

        backend = factory.create("gemini", config)

        assert isinstance(backend, GeminiBackend)
        assert backend.default_model == "gemini-2.5-flash"
        backend.close()

    def test_creates_gemini_backend_for_aiplatform_url(self, mock_lg: Logger) -> None:
        """Test factory creates GeminiBackend for AI Platform URLs."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://us-central1-aiplatform.googleapis.com/v1/projects/myproject/locations/us-central1/endpoints/openapi",
            }
        )

        backend = factory.create("gemini", config)

        assert isinstance(backend, GeminiBackend)
        backend.close()

    def test_creates_openai_backend_for_openai_url(self, mock_lg: Logger) -> None:
        """Test factory creates OpenAICompatibleBackend for OpenAI URLs."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
            }
        )

        backend = factory.create("openai", config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert not isinstance(backend, GeminiBackend)
        backend.close()

    def test_creates_openai_backend_for_local_url(self, mock_lg: Logger) -> None:
        """Test factory creates OpenAICompatibleBackend for local URLs."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
            }
        )

        backend = factory.create("local", config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert not isinstance(backend, GeminiBackend)
        backend.close()

    def test_creates_openai_backend_for_unknown_url(self, mock_lg: Logger) -> None:
        """Test factory creates OpenAICompatibleBackend for unknown URLs."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://some-custom-provider.example.com/v1",
            }
        )

        backend = factory.create("custom", config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert not isinstance(backend, GeminiBackend)
        backend.close()
