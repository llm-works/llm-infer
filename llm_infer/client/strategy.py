"""Routing strategy protocol and types.

Strategies determine which backends to try and in what order. The router
calls strategy.select() before each request and strategy.on_result() after.

Custom strategies can be loaded from external packages via StrategyFactory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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


class TransientAction(Enum):
    """Action to take after a transient error."""

    FAIL = "fail"  # Fatal error, raise immediately
    RETRY_SAME = "retry_same"  # Retry same backend with backoff (429 rate limit)
    RETRY_NEXT = "retry_next"  # Try different backend (5xx, timeout, unavailable)


class DecisionType(Enum):
    """Type of routing decision.

    Used by router to determine model resolution behavior.
    """

    INITIAL = "initial"  # First attempt, preserve requested model
    RETRY_SAME = "retry_same"  # Retry same backend, preserve model
    FALLBACK = "fallback"  # Different backend, use target's default model


# Status codes that should retry on a different backend
RETRY_NEXT_STATUS_CODES = frozenset({500, 502, 503, 504})

# Status codes that should retry same backend with backoff
RETRY_SAME_STATUS_CODES = frozenset({429})


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
        decision_type: Type of decision (initial, retry_same, fallback).
            Determines model resolution behavior.
    """

    backend: str
    updated_request: ChatRequest | None = None
    metadata: DotDict = field(default_factory=DotDict)
    decision_type: DecisionType = DecisionType.INITIAL


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
    """Protocol for classifying errors by retry action.

    Determines whether an error should fail immediately, retry the same
    backend (with backoff), or try a different backend.

    Implement custom detectors to handle application-specific error patterns.
    """

    def classify(self, error: BackendError) -> TransientAction:
        """Classify error to determine retry action.

        Args:
            error: The error that occurred.

        Returns:
            TransientAction indicating how to handle the error.
        """
        ...


class DefaultTransientDetector:
    """Default transient error classifier.

    Classification:
    - RETRY_NEXT: BackendUnavailableError, BackendTimeoutError, 5xx errors
    - RETRY_SAME: 429 rate limit (let client backoff handle it)
    - FAIL: All other errors (4xx client errors)
    """

    def classify(self, error: BackendError) -> TransientAction:
        """Classify error to determine retry action."""
        if isinstance(error, BackendUnavailableError | BackendTimeoutError):
            return TransientAction.RETRY_NEXT
        if isinstance(error, BackendRequestError) and error.status_code is not None:
            if error.status_code in RETRY_NEXT_STATUS_CODES:
                return TransientAction.RETRY_NEXT
            if error.status_code in RETRY_SAME_STATUS_CODES:
                return TransientAction.RETRY_SAME
        return TransientAction.FAIL


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
    module must export a `Factory` class implementing this protocol.

    Example config:
        strategy:
          factory: myapp.routing
          priority_order: [fast, reliable]

    Example module (myapp/routing.py):
        class Factory:
            def create(self, lg, config):
                return PriorityStrategy(
                    order=config.get("priority_order", []),
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
