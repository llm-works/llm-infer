"""Unit tests for BackendFactory."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from llm_infer.client.backends import BackendFactory
from llm_infer.client.backends.auth import (
    GCPServiceAccountAuth,
    StaticAPIKeyAuth,
)
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


class TestBackendFactoryAuth:
    """Test BackendFactory's auth: config block parsing."""

    @pytest.fixture
    def fake_sa_file(self, tmp_path: Path) -> str:
        p = tmp_path / "sa.json"
        p.write_text(json.dumps({"type": "service_account"}))
        return str(p)

    def test_top_level_api_key_wraps_as_static_auth(self, mock_lg: Logger) -> None:
        """Backwards compat: top-level api_key still works."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
            }
        )
        backend = factory.create("openai", config)
        assert isinstance(backend._auth, StaticAPIKeyAuth)
        assert backend._build_headers()["Authorization"] == "Bearer sk-test"
        backend.close()

    def test_auth_block_api_key_mode(self, mock_lg: Logger) -> None:
        """auth.mode=api_key with inline key."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "auth": {"mode": "api_key", "api_key": "sk-inline"},
            }
        )
        backend = factory.create("openai", config)
        assert backend._build_headers()["Authorization"] == "Bearer sk-inline"
        backend.close()

    def test_auth_block_gcp_sa_mode(self, mock_lg: Logger, fake_sa_file: str) -> None:
        """auth.mode=gcp_sa returns Gemini backend with GCP SA auth (Vertex)."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://aiplatform.googleapis.com/v1",
                "model": "google/gemini-2.5-flash",
                "auth": {"mode": "gcp_sa", "credentials_path": fake_sa_file},
            }
        )
        mock_creds = MagicMock()
        mock_creds.token = "tok"
        mock_creds.expiry = None
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            backend = factory.create("vertex", config)
            assert isinstance(backend, GeminiBackend)
            assert isinstance(backend._auth, GCPServiceAccountAuth)
            assert backend._build_headers()["Authorization"] == "Bearer tok"
            backend.close()

    def test_no_auth_no_api_key(self, mock_lg: Logger) -> None:
        """Local backends with no auth and no api_key produce no Authorization."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
            }
        )
        backend = factory.create("local", config)
        assert backend._auth is None
        assert "Authorization" not in backend._build_headers()
        backend.close()
