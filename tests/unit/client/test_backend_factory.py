"""Unit tests for BackendFactory."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from appinfra.yaml import SecretStr

from llm_infer.client.backends import BackendFactory, NativeVertexBackend
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

    def test_factory_passes_service_tier_to_gemini(self, mock_lg: Logger) -> None:
        """Factory threads config.service_tier into the GeminiBackend."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": (
                    "https://us-central1-aiplatform.googleapis.com/v1/projects/"
                    "p/locations/us-central1/endpoints/openapi"
                ),
                "service_tier": "priority",
            }
        )

        backend = factory.create("gemini", config)

        assert isinstance(backend, GeminiBackend)
        assert backend._service_tier == "priority"
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

    def test_explicit_provider_google_overrides_url(self, mock_lg: Logger) -> None:
        """Explicit provider=google routes to GeminiBackend even on a URL
        that would auto-detect differently. Mismatch is warned; explicit
        wins — and the backend.provider property returns the explicit value."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "provider": "google",
            }
        )

        backend = factory.create("proxied", config)

        assert isinstance(backend, GeminiBackend)
        # Explicit provider must be propagated to the backend instance
        assert backend.provider == "google"
        mock_lg.warning.assert_called_once()
        _, kwargs = mock_lg.warning.call_args
        assert kwargs["extra"]["backend"] == "proxied"
        assert kwargs["extra"]["provider"] == {
            "explicit": "google",
            "detected": "openai",
        }
        # base_url must not leak into the log — see factory._resolve_provider
        assert "base_url" not in kwargs["extra"]
        backend.close()

    def test_explicit_provider_matches_detected_no_warning(
        self, mock_lg: Logger
    ) -> None:
        """Explicit provider matching auto-detect stays silent."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "provider": "openai",
            }
        )

        backend = factory.create("openai", config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert not isinstance(backend, GeminiBackend)
        mock_lg.warning.assert_not_called()
        backend.close()

    def test_explicit_provider_on_unknown_url_no_warning(self, mock_lg: Logger) -> None:
        """Explicit provider on an unrecognized URL — no warning (UNKNOWN isn't
        a real disagreement), and the backend.provider returns the explicit value."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://some-custom-provider.example.com/v1",
                "provider": "openai",
            }
        )

        backend = factory.create("custom", config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert not isinstance(backend, GeminiBackend)
        # Explicit provider must be propagated — would be "unknown" if auto-detected
        assert backend.provider == "openai"
        mock_lg.warning.assert_not_called()
        backend.close()

    def test_explicit_provider_invalid_raises(self, mock_lg: Logger) -> None:
        """Invalid provider name raises ValueError listing valid options."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "provider": "nonexistent",
            }
        )

        with pytest.raises(ValueError, match="invalid provider 'nonexistent'"):
            factory.create("bad", config)


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

    def test_top_level_secret_api_key_wraps_as_static_auth(
        self, mock_lg: Logger
    ) -> None:
        """xray path: top-level api_key arrives as SecretStr; header still builds."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "api_key": SecretStr("sk-secret"),
            }
        )
        backend = factory.create("openai", config)
        assert isinstance(backend._auth, StaticAPIKeyAuth)
        assert backend._build_headers()["Authorization"] == "Bearer sk-secret"
        backend.close()

    def test_auth_block_secret_api_key_mode(self, mock_lg: Logger) -> None:
        """auth.mode=api_key with SecretStr inline key."""
        factory = BackendFactory(mock_lg)
        config = DotDict(
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "auth": {"mode": "api_key", "api_key": SecretStr("sk-inline-secret")},
            }
        )
        backend = factory.create("openai", config)
        assert backend._build_headers()["Authorization"] == "Bearer sk-inline-secret"
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


class TestBackendFactoryVertexNative:
    """``create_vertex_native`` reads ``project`` / ``region`` from yaml
    and delegates to ``NativeVertexFactory`` — sibling surface to the chat
    ``create()`` dispatch. ``create()`` itself rejects a ``type:
    vertex_native`` block pointing callers at the right accessor."""

    def test_reads_project_and_region_from_yaml(self, mock_lg: Logger) -> None:
        factory = BackendFactory(mock_lg)
        cfg = DotDict(
            {
                "type": "vertex_native",
                "project": "my-proj",
                "region": "us-central1",
                "auth": {"mode": "api_key", "api_key": "k"},
                "timeout": 45.0,
                "rate_limit": {"per_minute": 120},
                "service_tier": "priority",
            }
        )
        backend = factory.create_vertex_native("vertex_direct", cfg)
        assert isinstance(backend, NativeVertexBackend)
        assert backend._project == "my-proj"
        assert backend._region == "us-central1"
        assert backend._ctx.request_timeout == 45.0
        assert backend._service_tier == "priority"

    def test_missing_project_raises(self, mock_lg: Logger) -> None:
        factory = BackendFactory(mock_lg)
        cfg = DotDict(
            {
                "type": "vertex_native",
                "region": "us-central1",
                "auth": {"mode": "api_key", "api_key": "k"},
            }
        )
        with pytest.raises(ValueError, match="project.*region.*are required"):
            factory.create_vertex_native("vertex_direct", cfg)

    def test_missing_region_raises(self, mock_lg: Logger) -> None:
        factory = BackendFactory(mock_lg)
        cfg = DotDict(
            {
                "type": "vertex_native",
                "project": "my-proj",
                "auth": {"mode": "api_key", "api_key": "k"},
            }
        )
        with pytest.raises(ValueError, match="project.*region.*are required"):
            factory.create_vertex_native("vertex_direct", cfg)

    def test_missing_auth_raises(self, mock_lg: Logger) -> None:
        # NativeVertexFactory itself enforces this; verify create_vertex_native
        # propagates the failure at wire-up rather than swallowing it.
        factory = BackendFactory(mock_lg)
        cfg = DotDict(
            {
                "type": "vertex_native",
                "project": "my-proj",
                "region": "us-central1",
            }
        )
        with pytest.raises(ValueError, match="auth"):
            factory.create_vertex_native("vertex_direct", cfg)

    def test_create_rejects_vertex_native_type(self, mock_lg: Logger) -> None:
        # A yaml misfiring on the chat dispatch should point at the right
        # accessor rather than silently creating a broken chat backend.
        factory = BackendFactory(mock_lg)
        cfg = DotDict(
            {
                "type": "vertex_native",
                "project": "my-proj",
                "region": "us-central1",
                "auth": {"mode": "api_key", "api_key": "k"},
            }
        )
        with pytest.raises(ValueError, match="vertex_natives_from_config"):
            factory.create("vertex_direct", cfg)
