"""Routing strategy protocol and types.

Strategies determine which backends to try and in what order. The router
calls strategy.select() before each request and strategy.on_result() after.

Custom strategies can be loaded from external packages via StrategyFactory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from .errors import (
    BackendError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from .types import ChatRequest, ChatResponse

if TYPE_CHECKING:
    from .router import LLMRouter

# Status codes considered transient (worth retrying on fallback)
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass
class RoutingContext:
    """Context for routing decisions.

    Passed to strategy.select() so it can make informed decisions.

    Attributes:
        request: The ChatRequest being routed.
        backend: Explicit backend requested (highest priority).
        role: Application-defined role (e.g., "exploration", "synthesis").
        metadata: Extensible DotDict for user data and strategy state.
    """

    request: ChatRequest
    backend: str | None = None
    role: str | None = None
    metadata: DotDict = field(default_factory=DotDict)


@dataclass
class RoutingDecision:
    """Result from strategy.select() or on_error().

    Attributes:
        backend: Backend name to try.
        updated_request: Optional modified request. If set, overrides the
            original ChatRequest for this attempt.
        metadata: Decision metadata (reason, markers, etc.).
    """

    backend: str
    updated_request: ChatRequest | None = None
    metadata: DotDict = field(default_factory=DotDict)


@dataclass
class RoutingResult:
    """Result of a request attempt, passed to on_error() and on_result().

    Contains everything the strategy needs to make decisions.

    Attributes:
        backend: Backend that handled the request.
        model: Actual model used (from response).
        context: Original routing context (role, user data, state).
        decision: The decision that led to this attempt.
        response: Full ChatResponse if successful.
        error: Error if failed, None if success.
        metadata: Result metadata (latency_ms, etc.).

    The request sent is: decision.updated_request or context.request
    """

    backend: str
    model: str | None
    context: RoutingContext
    decision: RoutingDecision
    response: ChatResponse | None = None
    error: BackendError | None = None
    metadata: DotDict = field(default_factory=DotDict)


@runtime_checkable
class TransientDetector(Protocol):
    """Protocol for classifying errors as transient or fatal.

    Transient errors trigger fallback to the next backend. Fatal errors
    are raised immediately without trying other backends.

    Implement custom detectors to handle application-specific error patterns.
    """

    def is_transient(self, error: BackendError) -> bool:
        """Check if error should trigger fallback to next backend.

        Args:
            error: The error that occurred.

        Returns:
            True if router should try the next backend, False to raise immediately.
        """
        ...


class DefaultTransientDetector:
    """Default transient error detector.

    Considers these errors transient (worth retrying on fallback):
    - BackendUnavailableError (connection failed)
    - BackendTimeoutError (request timed out)
    - BackendRequestError with status 429, 500, 502, 503, 504
    """

    def is_transient(self, error: BackendError) -> bool:
        """Check if error is transient."""
        if isinstance(error, BackendUnavailableError | BackendTimeoutError):
            return True
        if isinstance(error, BackendRequestError):
            return (
                error.status_code is not None
                and error.status_code in TRANSIENT_STATUS_CODES
            )
        return False


@runtime_checkable
class RoutingStrategy(Protocol):
    """Protocol for routing strategies.

    Strategies control the full request lifecycle:
    - select(): Pick first backend
    - on_error(): Decide retry after failure
    - on_result(): Feedback on completion
    """

    def select(
        self, router: LLMRouter, context: RoutingContext
    ) -> RoutingDecision | None:
        """Pick backend for initial attempt.

        Args:
            router: The router (access to clients, config).
            context: Request context (request, backend hint, role, metadata).

        Returns:
            RoutingDecision with backend and optional modified request.
            None means use router's default resolution.
        """
        ...

    def on_error(
        self, router: LLMRouter, result: RoutingResult
    ) -> RoutingDecision | None:
        """Decide next action after a failed attempt.

        Args:
            router: The router (access to clients, config).
            result: Complete attempt details including error, routing_context,
                and decision that led here.

        Returns:
            RoutingDecision to retry with new backend/request.
            None to stop retrying and raise the error.
        """
        ...

    def on_result(self, router: LLMRouter, result: RoutingResult) -> None:
        """Feedback after successful completion.

        Called after a request succeeds. Use for tracking latency, health,
        token usage, etc. Should be fast and non-blocking.

        Args:
            router: The router that made the request.
            result: Success details (backend, response, latency, tokens).
        """
        ...


@runtime_checkable
class StrategyFactory(Protocol):
    """Protocol for strategy factories.

    Allows external packages to provide custom routing strategies. The factory
    is loaded from a Python module path specified in config, then called with
    logger and config to create the strategy.

    Example config:
        strategy:
          factory: appware.billing:BudgetStrategyFactory
          budget_limit: 100
          fallback_order: [gemini, grok, openai]

    Example factory:
        class BudgetStrategyFactory:
            def create(self, lg, config):
                return BudgetAwareStrategy(
                    budget_limit=config.get("budget_limit", 100),
                    fallback_order=config.get("fallback_order", []),
                )
    """

    def create(
        self,
        lg: Logger,
        config: DotDict,
    ) -> RoutingStrategy:
        """Create a routing strategy.

        Args:
            lg: Logger instance.
            config: Strategy configuration from DotDict.

        Returns:
            Configured routing strategy.
        """
        ...
