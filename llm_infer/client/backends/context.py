"""Backend configuration and context."""

from __future__ import annotations

from dataclasses import dataclass

from appinfra.rate_limit import RateLimiter


@dataclass
class RetryConfig:
    """Retry configuration (stateless)."""

    base: float = 1.0
    factor: float = 2.0
    max_delay: float = 60.0
    timeout: float = 0


@dataclass
class BackendContext:
    """Shared context for backend behavior.

    Created by Factory from config, passed to Backend.
    """

    rate_limiter: RateLimiter | None = None
    retry: RetryConfig | None = None
    request_timeout: float = 120.0
