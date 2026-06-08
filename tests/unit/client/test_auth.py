"""Unit tests for backend auth providers."""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from appinfra.log import Logger

from llm_infer.client.backends.auth import (
    AuthProvider,
    GCPServiceAccountAuth,
    GoogleAPIKeyHeaderAuth,
    StaticAPIKeyAuth,
    auth_from_api_key,
    auth_from_config,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


@pytest.fixture
def fake_sa_file(tmp_path: Path) -> str:
    """Write a placeholder SA JSON file. Contents are irrelevant — we mock the
    Credentials loader; only the path's existence matters."""
    p = tmp_path / "sa.json"
    p.write_text(json.dumps({"type": "service_account", "private_key": "fake"}))
    return str(p)


class TestStaticAPIKeyAuth:
    def test_headers(self) -> None:
        auth = StaticAPIKeyAuth("sk-test")
        assert auth.headers() == {"Authorization": "Bearer sk-test"}

    def test_headers_async(self) -> None:
        auth = StaticAPIKeyAuth("sk-test")
        result = asyncio.run(auth.headers_async())
        assert result == {"Authorization": "Bearer sk-test"}

    def test_implements_protocol(self) -> None:
        auth = StaticAPIKeyAuth("k")
        assert isinstance(auth, AuthProvider)


class TestGoogleAPIKeyHeaderAuth:
    def test_headers(self) -> None:
        auth = GoogleAPIKeyHeaderAuth("AIza-key")
        assert auth.headers() == {"x-goog-api-key": "AIza-key"}

    def test_headers_async(self) -> None:
        auth = GoogleAPIKeyHeaderAuth("AIza-key")
        assert asyncio.run(auth.headers_async()) == {"x-goog-api-key": "AIza-key"}


class TestAuthFromAPIKey:
    def test_none_returns_none(self) -> None:
        assert auth_from_api_key(None) is None

    def test_default_header_returns_bearer(self) -> None:
        auth = auth_from_api_key("sk-abc")
        assert isinstance(auth, StaticAPIKeyAuth)
        assert auth.headers() == {"Authorization": "Bearer sk-abc"}

    def test_google_header_returns_xgoog(self) -> None:
        auth = auth_from_api_key("AIza-abc", header="x-goog-api-key")
        assert isinstance(auth, GoogleAPIKeyHeaderAuth)
        assert auth.headers() == {"x-goog-api-key": "AIza-abc"}


def _mock_creds(token: str = "tok-1", expiry_offset_s: int = 3600) -> MagicMock:
    """Build a mock google-auth Credentials object.

    google-auth uses naive UTC datetimes for `expiry`, so we mirror that here
    to exercise the same tzinfo path as production credentials.
    """
    creds = MagicMock()
    creds.token = token
    now_naive_utc = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    creds.expiry = now_naive_utc + datetime.timedelta(seconds=expiry_offset_s)

    def _do_refresh(_request: object) -> None:
        creds.token = f"{token}-refreshed"
        creds.expiry = now_naive_utc + datetime.timedelta(seconds=3600)

    creds.refresh.side_effect = _do_refresh
    return creds


class TestGCPServiceAccountAuth:
    def test_raises_when_path_missing(self, mock_lg: Logger, monkeypatch) -> None:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        with pytest.raises(ValueError, match="No credentials path"):
            GCPServiceAccountAuth(mock_lg)

    def test_raises_when_file_does_not_exist(self, mock_lg: Logger) -> None:
        with pytest.raises(FileNotFoundError):
            GCPServiceAccountAuth(mock_lg, credentials_path="/nonexistent/sa.json")

    def test_loads_from_env_var(
        self, mock_lg: Logger, fake_sa_file: str, monkeypatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", fake_sa_file)
        mock_creds = _mock_creds()
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ) as loader:
            GCPServiceAccountAuth(mock_lg)
            loader.assert_called_once()
            args, kwargs = loader.call_args
            assert args[0] == fake_sa_file
            assert kwargs["scopes"] == [
                "https://www.googleapis.com/auth/cloud-platform"
            ]

    def test_custom_scopes(self, mock_lg: Logger, fake_sa_file: str) -> None:
        mock_creds = _mock_creds()
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ) as loader:
            GCPServiceAccountAuth(
                mock_lg,
                credentials_path=fake_sa_file,
                scopes=["https://www.googleapis.com/auth/scope-a"],
            )
            _args, kwargs = loader.call_args
            assert kwargs["scopes"] == ["https://www.googleapis.com/auth/scope-a"]

    def test_headers_returns_bearer(self, mock_lg: Logger, fake_sa_file: str) -> None:
        mock_creds = _mock_creds(token="abc")
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(mock_lg, credentials_path=fake_sa_file)
            assert auth.headers() == {"Authorization": "Bearer abc"}

    def test_refresh_when_token_none(self, mock_lg: Logger, fake_sa_file: str) -> None:
        mock_creds = _mock_creds(token="abc")
        mock_creds.token = None  # force initial refresh
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(mock_lg, credentials_path=fake_sa_file)
            auth.headers()
            mock_creds.refresh.assert_called_once()

    def test_refresh_when_near_expiry(self, mock_lg: Logger, fake_sa_file: str) -> None:
        # Token expires in 60s with default skew of 300s → must refresh
        mock_creds = _mock_creds(token="stale", expiry_offset_s=60)
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(mock_lg, credentials_path=fake_sa_file)
            headers = auth.headers()
            mock_creds.refresh.assert_called_once()
            assert headers == {"Authorization": "Bearer stale-refreshed"}

    def test_no_refresh_when_token_fresh(
        self, mock_lg: Logger, fake_sa_file: str
    ) -> None:
        # Token expires in 3600s, skew is 300s → no refresh
        mock_creds = _mock_creds(token="fresh", expiry_offset_s=3600)
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(mock_lg, credentials_path=fake_sa_file)
            auth.headers()
            auth.headers()
            mock_creds.refresh.assert_not_called()

    def test_no_refresh_when_expiry_unset(
        self, mock_lg: Logger, fake_sa_file: str
    ) -> None:
        # token present, expiry None → cannot determine → don't refresh
        mock_creds = _mock_creds(token="x")
        mock_creds.expiry = None
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(mock_lg, credentials_path=fake_sa_file)
            auth.headers()
            mock_creds.refresh.assert_not_called()

    def test_refresh_skew_configurable(
        self, mock_lg: Logger, fake_sa_file: str
    ) -> None:
        # Token expires in 60s, but skew is 30s → no refresh
        mock_creds = _mock_creds(token="ok", expiry_offset_s=60)
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(
                mock_lg, credentials_path=fake_sa_file, refresh_skew_s=30
            )
            auth.headers()
            mock_creds.refresh.assert_not_called()

    def test_refresh_error_propagates_and_logs(
        self, mock_lg: Logger, fake_sa_file: str
    ) -> None:
        mock_creds = _mock_creds(token="x", expiry_offset_s=60)
        mock_creds.refresh.side_effect = RuntimeError("boom")
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(mock_lg, credentials_path=fake_sa_file)
            with pytest.raises(RuntimeError, match="boom"):
                auth.headers()
            mock_lg.warning.assert_called_once()

    def test_headers_async_offloads(self, mock_lg: Logger, fake_sa_file: str) -> None:
        mock_creds = _mock_creds(token="abc")
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(mock_lg, credentials_path=fake_sa_file)
            result = asyncio.run(auth.headers_async())
            assert result == {"Authorization": "Bearer abc"}


class TestAuthFromConfig:
    def test_none_with_api_key(self, mock_lg: Logger) -> None:
        auth = auth_from_config(mock_lg, None, api_key="k")
        assert isinstance(auth, StaticAPIKeyAuth)

    def test_none_without_api_key(self, mock_lg: Logger) -> None:
        assert auth_from_config(mock_lg, None) is None

    def test_api_key_mode(self, mock_lg: Logger) -> None:
        auth = auth_from_config(mock_lg, {"mode": "api_key", "api_key": "k1"})
        assert isinstance(auth, StaticAPIKeyAuth)
        assert auth.headers() == {"Authorization": "Bearer k1"}

    def test_api_key_mode_inline_key_wins(self, mock_lg: Logger) -> None:
        auth = auth_from_config(
            mock_lg, {"mode": "api_key", "api_key": "inline"}, api_key="fallback"
        )
        assert auth is not None and auth.headers()["Authorization"] == "Bearer inline"

    def test_api_key_mode_fallback_to_top_level(self, mock_lg: Logger) -> None:
        auth = auth_from_config(mock_lg, {"mode": "api_key"}, api_key="fallback")
        assert auth is not None and auth.headers() == {
            "Authorization": "Bearer fallback"
        }

    def test_api_key_header_override(self, mock_lg: Logger) -> None:
        auth = auth_from_config(
            mock_lg,
            {"mode": "api_key", "api_key": "AIz"},
            api_key_header="x-goog-api-key",
        )
        assert isinstance(auth, GoogleAPIKeyHeaderAuth)

    def test_gcp_sa_mode(self, mock_lg: Logger, fake_sa_file: str) -> None:
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=_mock_creds(),
        ) as loader:
            auth = auth_from_config(
                mock_lg,
                {"mode": "gcp_sa", "credentials_path": fake_sa_file},
            )
            assert isinstance(auth, GCPServiceAccountAuth)
            loader.assert_called_once()

    def test_gcp_sa_with_custom_options(
        self, mock_lg: Logger, fake_sa_file: str
    ) -> None:
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=_mock_creds(),
        ) as loader:
            auth_from_config(
                mock_lg,
                {
                    "mode": "gcp_sa",
                    "credentials_path": fake_sa_file,
                    "scopes": ["https://www.googleapis.com/auth/scope-x"],
                    "refresh_skew_s": 30,
                },
            )
            _args, kwargs = loader.call_args
            assert kwargs["scopes"] == ["https://www.googleapis.com/auth/scope-x"]

    def test_unknown_mode_raises(self, mock_lg: Logger) -> None:
        with pytest.raises(ValueError, match="Unknown auth mode"):
            auth_from_config(mock_lg, {"mode": "magic"})

    def test_gcp_sa_rejects_bare_string_scopes(self, mock_lg: Logger) -> None:
        with pytest.raises(ValueError, match="scopes must be a list"):
            auth_from_config(
                mock_lg,
                {"mode": "gcp_sa", "scopes": "https://example.com/scope"},
            )

    def test_gcp_sa_rejects_negative_refresh_skew(self, mock_lg: Logger) -> None:
        with pytest.raises(ValueError, match="refresh_skew_s must be"):
            auth_from_config(
                mock_lg,
                {"mode": "gcp_sa", "refresh_skew_s": -1},
            )

    def test_gcp_sa_rejects_non_int_refresh_skew(self, mock_lg: Logger) -> None:
        with pytest.raises(ValueError, match="refresh_skew_s must be"):
            auth_from_config(
                mock_lg,
                {"mode": "gcp_sa", "refresh_skew_s": "300"},
            )
