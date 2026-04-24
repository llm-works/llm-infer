"""Built-in routing strategies and factories.

Provides DefaultStrategy and FallbackStrategy. Custom strategies can be
implemented by following the RoutingStrategy protocol, and loaded via
StrategyFactory from external packages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from .strategy import (
    DefaultTransientDetector,
    RoutingContext,
    RoutingDecision,
    RoutingResult,
    TransientAction,
    TransientDetector,
)

if TYPE_CHECKING:
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


class FallbackStrategy(DefaultStrategy):
    """Fallback strategy - try backends in order until one succeeds.

    On transient errors (429, 5xx, timeout), tries the next backend.
    Fatal errors (4xx except 429) are raised immediately.

    Supports role-based routing: different backend orders for different roles.

    Example config:
        strategy:
          type: fallback
          order: [gemini, grok, openai]
          roles:
            synthesis: [gpt4, claude]
            exploration: [gemini, grok]
    """

    def __init__(
        self,
        order: list[str] | None = None,
        roles: dict[str, list[str]] | None = None,
        detector: TransientDetector | None = None,
    ) -> None:
        """Initialize with fallback configuration.

        Args:
            order: Ordered list of backends to try.
            roles: Optional role -> backend list mapping for role-specific
                routing (e.g., {"synthesis": ["gpt4", "claude"]}).
            detector: Custom transient error detector.
        """
        self._order = list(order) if order else []
        self._roles = roles or {}
        self._detector = detector or DefaultTransientDetector()

    def _get_order(self, context: RoutingContext) -> list[str]:
        """Get backend order based on role."""
        if context.role and context.role in self._roles:
            return self._roles[context.role]
        return self._order

    def _next_backend(
        self, order: list[str], tried: list[str], clients: Mapping[str, object]
    ) -> str | None:
        """Find next available backend not yet tried."""
        for backend in order:
            if backend not in tried and backend in clients:
                return backend
        return None

    def select(
        self, router: LLMRouter, context: RoutingContext
    ) -> RoutingDecision | None:
        """Pick first backend from order."""
        # Explicit backend - no fallback
        if context.backend is not None:
            return super().select(router, context)

        order = self._get_order(context)
        backend = self._next_backend(order, [], router.clients)
        if backend is None:
            return None

        # Store order and tried list in context for on_error
        context.metadata._fallback_order = order
        context.metadata._fallback_tried = [backend]
        return RoutingDecision(backend=backend)

    def on_error(
        self, router: LLMRouter, result: RoutingResult
    ) -> RoutingDecision | None:
        """Handle errors based on classification."""
        if not result.error:
            return None

        action = self._detector.classify(result.error)

        if action == TransientAction.FAIL:
            return None

        if action == TransientAction.RETRY_SAME:
            return RoutingDecision(
                backend=result.decision.backend,
                metadata=DotDict(reason="retry_same", error_code=429),
            )

        # RETRY_NEXT: try next backend
        ctx = result.context.metadata
        order = getattr(ctx, "_fallback_order", self._order)
        tried = getattr(ctx, "_fallback_tried", [])

        backend = self._next_backend(order, tried, router.clients)
        if backend is None:
            return None

        tried.append(backend)
        return RoutingDecision(
            backend=backend,
            metadata=DotDict(reason="fallback", previous=result.decision.backend),
        )


class FallbackStrategyFactory:
    """Factory for FallbackStrategy."""

    def create(
        self,
        lg: Logger,
        config: DotDict,
    ) -> FallbackStrategy:
        """Create FallbackStrategy from config.

        Config keys:
            order: list[str] - Ordered list of backend names
            roles: dict[str, list[str]] - Role-specific backend orders
        """
        order = config.get("order")
        roles = config.get("roles")
        return FallbackStrategy(
            order=list(order) if order else None,
            roles=dict(roles) if roles else None,
        )
