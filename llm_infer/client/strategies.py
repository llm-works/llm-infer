"""Built-in routing strategies and factories.

Provides DefaultStrategy for simple single-backend routing. Custom strategies
can be implemented by following the RoutingStrategy protocol, and loaded via
StrategyFactory from external packages.

For fallback/resilience behavior, use FallbackClient which wraps an LLMRouter
and provides model-to-model fallbacks with proper transient error handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from appinfra.log import Logger

from .strategy import (
    RoutingContext,
    RoutingDecision,
    RoutingResult,
)

if TYPE_CHECKING:
    from appinfra.dot_dict import DotDict

    from .router import LLMRouter


class DefaultStrategy:
    """Default routing strategy - single backend, no fallback.

    Returns explicit backend if specified, otherwise None (router uses default).
    Does not retry on errors.
    """

    def select(
        self, router: LLMRouter, context: RoutingContext
    ) -> RoutingDecision | None:
        """Select single backend based on context."""
        if context.backend is not None:
            return RoutingDecision(backend=context.backend)
        return None

    def on_error(
        self, router: LLMRouter, result: RoutingResult
    ) -> RoutingDecision | None:
        """No retry - default strategy doesn't fallback."""
        return None

    def on_result(self, router: LLMRouter, result: RoutingResult) -> None:
        """No-op - default strategy doesn't track results."""


class DefaultStrategyFactory:
    """Factory for DefaultStrategy."""

    def create(
        self,
        lg: Logger,
        config: DotDict,
    ) -> DefaultStrategy:
        """Create DefaultStrategy from config."""
        return DefaultStrategy()
