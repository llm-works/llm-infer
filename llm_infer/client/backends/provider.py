"""Provider identification for LLM backends.

This module provides the Provider enum and detection logic to identify
which LLM provider a backend is connecting to, based on URL patterns
and API key formats.
"""

from __future__ import annotations

from enum import StrEnum


class Provider(StrEnum):
    """Known LLM providers.

    Used to identify the actual provider for a backend, which determines
    the raw response format and provider-specific features like caching.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    XAI = "xai"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"
    UNKNOWN = "unknown"


class ProviderDetector:
    """Detects the LLM provider from backend configuration.

    Detection is based on URL patterns and API key formats. This allows
    accurate provider identification even when users give backends
    arbitrary names.
    """

    # URL patterns for provider detection (order matters - more specific first)
    _URL_PATTERNS: list[tuple[str, Provider]] = [
        ("api.anthropic.com", Provider.ANTHROPIC),
        ("api.openai.com", Provider.OPENAI),
        ("api.x.ai", Provider.XAI),
        ("generativelanguage.googleapis.com", Provider.GOOGLE),
        ("aiplatform.googleapis.com", Provider.GOOGLE),
        ("openai.azure.com", Provider.AZURE),
        ("localhost", Provider.LOCAL),
        ("127.0.0.1", Provider.LOCAL),
        ("0.0.0.0", Provider.LOCAL),
    ]

    # API key prefix patterns
    _KEY_PATTERNS: list[tuple[str, Provider]] = [
        ("sk-ant-", Provider.ANTHROPIC),
        # OpenAI keys start with sk- but so do many others, so not reliable
    ]

    @classmethod
    def detect(
        cls,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> Provider:
        """Detect provider from URL and/or API key.

        Args:
            base_url: The API base URL.
            api_key: The API key (used for prefix-based detection).

        Returns:
            The detected Provider, or Provider.UNKNOWN if not detected.
        """
        # URL patterns are most reliable
        if base_url:
            url_lower = base_url.lower()
            for pattern, provider in cls._URL_PATTERNS:
                if pattern in url_lower:
                    return provider

        # Fall back to API key patterns
        if api_key:
            for prefix, provider in cls._KEY_PATTERNS:
                if api_key.startswith(prefix):
                    return provider

        return Provider.UNKNOWN
