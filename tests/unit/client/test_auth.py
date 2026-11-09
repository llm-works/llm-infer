# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for backend auth providers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from appinfra import Envelope
from appinfra.log import Logger

from llm_infer.client.backends.auth import (
    AuthProvider,
    GoogleAPIKeyHeaderAuth,
    StaticAPIKeyAuth,
    auth_from_api_key,
    auth_from_config,
)


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


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

    def test_accepts_envelope(self) -> None:
        auth = StaticAPIKeyAuth(Envelope("sk-test"))
        assert auth.headers() == {"Authorization": "Bearer sk-test"}

    def test_stores_as_envelope(self) -> None:
        """The stored key must be an Envelope — this is what keeps the value
        out of tracebacks and Logger.extra, and is the whole point of the
        migration. If storage regresses to plain str, this fails."""
        auth = StaticAPIKeyAuth("sk-repr-safe")
        assert isinstance(auth._api_key, Envelope)
        assert "sk-repr-safe" not in repr(auth._api_key)
        assert "sk-repr-safe" not in str(auth._api_key)


class TestGoogleAPIKeyHeaderAuth:
    def test_headers(self) -> None:
        auth = GoogleAPIKeyHeaderAuth("AIza-key")
        assert auth.headers() == {"x-goog-api-key": "AIza-key"}

    def test_headers_async(self) -> None:
        auth = GoogleAPIKeyHeaderAuth("AIza-key")
        assert asyncio.run(auth.headers_async()) == {"x-goog-api-key": "AIza-key"}

    def test_accepts_envelope(self) -> None:
        auth = GoogleAPIKeyHeaderAuth(Envelope("AIza-key"))
        assert auth.headers() == {"x-goog-api-key": "AIza-key"}

    def test_stores_as_envelope(self) -> None:
        auth = GoogleAPIKeyHeaderAuth("AIza-repr-safe")
        assert isinstance(auth._api_key, Envelope)
        assert "AIza-repr-safe" not in repr(auth._api_key)


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

    def test_accepts_envelope(self) -> None:
        auth = auth_from_api_key(Envelope("sk-abc"))
        assert isinstance(auth, StaticAPIKeyAuth)
        assert auth.headers() == {"Authorization": "Bearer sk-abc"}

    def test_accepts_envelope_with_google_header(self) -> None:
        auth = auth_from_api_key(Envelope("AIza-abc"), header="x-goog-api-key")
        assert isinstance(auth, GoogleAPIKeyHeaderAuth)
        assert auth.headers() == {"x-goog-api-key": "AIza-abc"}


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
