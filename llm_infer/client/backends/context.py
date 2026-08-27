# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Backend configuration and context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from appinfra.log import Logger
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


def context_from_config(lg: Logger, cfg: dict[str, Any]) -> BackendContext:
    """Build a BackendContext from a backend config dict.

    Shared by BackendFactory and NativeVertexFactory so config parsing
    stays in sync (same defaults, same keys). Accepts plain dict or
    DotDict — both support ``.get()``.
    """
    return BackendContext(
        rate_limiter=_rate_limiter_from_config(lg, cfg.get("rate_limit")),
        retry=_retry_from_config(cfg.get("retry")),
        request_timeout=float(cfg.get("timeout", 120.0)),
    )


def _rate_limiter_from_config(
    lg: Logger, cfg: dict[str, Any] | None
) -> RateLimiter | None:
    if not cfg:
        return None
    return RateLimiter(lg, per_minute=cfg.get("per_minute", 60))


def _retry_from_config(cfg: dict[str, Any] | None) -> RetryConfig | None:
    if not cfg:
        return None
    return RetryConfig(
        base=cfg.get("base", 1.0),
        factor=cfg.get("factor", 2.0),
        max_delay=cfg.get("max_delay", 60.0),
        timeout=cfg.get("timeout", 0),
    )
