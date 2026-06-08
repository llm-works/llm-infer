"""Authentication providers for backend HTTP requests.

Backends call the provider to obtain request headers. Providers handle their
own caching and refresh; the backend never sees raw credentials.

Three implementations cover current providers:

- ``StaticAPIKeyAuth``: ``Authorization: Bearer <key>`` (OpenAI, xAI, Anthropic
  static key, AI Studio chat).
- ``GoogleAPIKeyHeaderAuth``: ``x-goog-api-key: <key>`` (AI Studio embeddings).
- ``GCPServiceAccountAuth``: ``Authorization: Bearer <token>`` with token
  exchange + refresh from a service-account JSON key (Vertex AI). Requires the
  ``[gcp]`` extra (``pip install llm-infer[gcp]``).

The async path wraps the sync refresh in ``asyncio.to_thread`` because
``google-auth``'s transport is sync-only.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from appinfra.log import Logger

if TYPE_CHECKING:
    from google.oauth2.service_account import Credentials


_DEFAULT_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
_DEFAULT_REFRESH_SKEW_S = 300


@runtime_checkable
class AuthProvider(Protocol):
    """Supplies HTTP auth headers, refreshing credentials when needed."""

    def headers(self) -> dict[str, str]:
        """Return auth headers for a sync request."""
        ...

    async def headers_async(self) -> dict[str, str]:
        """Return auth headers for an async request."""
        ...


class StaticAPIKeyAuth:
    """``Authorization: Bearer <key>``."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def headers_async(self) -> dict[str, str]:
        return self.headers()


class GoogleAPIKeyHeaderAuth:
    """``x-goog-api-key: <key>`` (AI Studio embeddings)."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    async def headers_async(self) -> dict[str, str]:
        return self.headers()


class GCPServiceAccountAuth:
    """GCP service-account OAuth bearer auth.

    Loads a service-account key (JSON file path, contents, or the
    ``GOOGLE_APPLICATION_CREDENTIALS`` env var) and exchanges it for a
    short-lived access token, refreshing before expiry. Lazy-imports
    ``google.auth`` so projects that don't need Vertex aren't forced to
    install ``google-auth``.
    """

    def __init__(
        self,
        lg: Logger,
        credentials_path: str | None = None,
        scopes: list[str] | tuple[str, ...] | None = None,
        refresh_skew_s: int = _DEFAULT_REFRESH_SKEW_S,
    ) -> None:
        self._lg = lg
        self._refresh_skew_s = refresh_skew_s
        self._lock = threading.Lock()
        self._creds, self._request = self._load_credentials(credentials_path, scopes)

    @staticmethod
    def _load_credentials(
        credentials_path: str | None,
        scopes: list[str] | tuple[str, ...] | None,
    ) -> tuple[Credentials, Any]:
        """Resolve the SA key path and load credentials.

        Lazy-imports google-auth so projects without the [gcp] extra aren't
        forced to install it.
        """
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
        except ImportError as e:
            raise ImportError(
                "GCPServiceAccountAuth requires google-auth. "
                "Install with: pip install 'llm-infer[gcp]'"
            ) from e

        path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not path:
            raise ValueError(
                "No credentials path provided and GOOGLE_APPLICATION_CREDENTIALS "
                "is not set"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Service account key not found: {path}")

        effective_scopes = list(scopes) if scopes else list(_DEFAULT_SCOPES)
        creds = service_account.Credentials.from_service_account_file(
            path, scopes=effective_scopes
        )
        return creds, Request()

    def headers(self) -> dict[str, str]:
        token = self._get_token()
        return {"Authorization": f"Bearer {token}"}

    async def headers_async(self) -> dict[str, str]:
        return await asyncio.to_thread(self.headers)

    def _get_token(self) -> str:
        """Return a valid token, refreshing if needed.

        Reads the token under the lock to avoid racing with a concurrent
        refresh on another thread.
        """
        with self._lock:
            if self._needs_refresh():
                try:
                    self._creds.refresh(self._request)
                except Exception as e:
                    self._lg.warning(
                        "GCP service account token refresh failed",
                        extra={"exception": e},
                    )
                    raise
            return self._creds.token  # type: ignore[no-any-return]

    def _needs_refresh(self) -> bool:
        if self._creds.token is None:
            return True
        expiry = self._creds.expiry
        if expiry is None:
            return False
        # google-auth populates `expiry` as a naive UTC datetime. Attach UTC
        # tzinfo explicitly — otherwise `.timestamp()` would apply the local
        # offset and the comparison breaks on non-UTC machines.
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=datetime.UTC)
        return bool(expiry.timestamp() - time.time() < self._refresh_skew_s)


def auth_from_api_key(
    api_key: str | None, *, header: str = "Authorization"
) -> AuthProvider | None:
    """Wrap an API key in the right AuthProvider, or return None if no key.

    ``header='x-goog-api-key'`` yields ``GoogleAPIKeyHeaderAuth`` (AI Studio
    embeddings); anything else yields ``StaticAPIKeyAuth`` (Bearer).
    """
    if api_key is None:
        return None
    if header == "x-goog-api-key":
        return GoogleAPIKeyHeaderAuth(api_key)
    return StaticAPIKeyAuth(api_key)


def auth_from_config(
    lg: Logger,
    auth_cfg: dict[str, Any] | None,
    *,
    api_key: str | None = None,
    api_key_header: str = "Authorization",
) -> AuthProvider | None:
    """Build an ``AuthProvider`` from an ``auth:`` config block.

    If ``auth_cfg`` is None, falls back to wrapping ``api_key`` (backwards
    compatibility for configs that put the key at the top level).

    Supported modes:
        - ``api_key`` (default): static bearer (or ``x-goog-api-key`` if
          ``api_key_header='x-goog-api-key'``). Reads ``api_key`` from the
          block or the fallback parameter.
        - ``gcp_sa``: ``GCPServiceAccountAuth``. Optional fields:
          ``credentials_path``, ``scopes``, ``refresh_skew_s``.

    Args:
        lg: Logger.
        auth_cfg: The ``auth:`` block, or None.
        api_key: Fallback API key (top-level config) used when ``auth_cfg`` is
            None or its mode is ``api_key`` without an inline key.
        api_key_header: Header name when wrapping ``api_key`` statically.

    Raises:
        ValueError: Unknown ``mode``.
    """
    if auth_cfg is None:
        return auth_from_api_key(api_key, header=api_key_header)

    mode = auth_cfg.get("mode", "api_key")
    if mode == "api_key":
        key = auth_cfg.get("api_key", api_key)
        return auth_from_api_key(key, header=api_key_header)
    if mode == "gcp_sa":
        scopes = auth_cfg.get("scopes")
        return GCPServiceAccountAuth(
            lg,
            credentials_path=auth_cfg.get("credentials_path"),
            scopes=list(scopes) if scopes else None,
            refresh_skew_s=auth_cfg.get("refresh_skew_s", _DEFAULT_REFRESH_SKEW_S),
        )
    raise ValueError(f"Unknown auth mode: {mode!r}")


__all__ = [
    "AuthProvider",
    "GCPServiceAccountAuth",
    "GoogleAPIKeyHeaderAuth",
    "StaticAPIKeyAuth",
    "auth_from_api_key",
    "auth_from_config",
]
