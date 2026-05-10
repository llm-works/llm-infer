"""Tests for routing strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client.errors import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_infer.client.strategies import (
    DefaultStrategy,
    DefaultStrategyFactory,
    FallbackStrategy,
    FallbackStrategyFactory,
)
from llm_infer.client.strategy import (
    DefaultTransientDetector,
    RoutingContext,
    RoutingDecision,
    RoutingResult,
    TransientAction,
)
from llm_infer.client.types import ChatRequest

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_router() -> MagicMock:
    """Create mock router with clients."""
    router = MagicMock()
    router.clients = {"primary": MagicMock(), "fallback": MagicMock()}
    return router


@pytest.fixture
def mock_request() -> ChatRequest:
    """Create mock chat request."""
    return ChatRequest(messages=[{"role": "user", "content": "hello"}])


class TestDefaultTransientDetector:
    """Tests for DefaultTransientDetector."""

    def test_unavailable_is_retry_next(self) -> None:
        """BackendUnavailableError should retry next backend."""
        detector = DefaultTransientDetector()
        error = BackendUnavailableError("connection failed")
        assert detector.classify(error) == TransientAction.RETRY_NEXT

    def test_timeout_is_retry_next(self) -> None:
        """BackendTimeoutError should retry next backend."""
        detector = DefaultTransientDetector()
        error = BackendTimeoutError("timed out")
        assert detector.classify(error) == TransientAction.RETRY_NEXT

    def test_429_is_retry_same(self) -> None:
        """429 rate limit should retry same backend."""
        detector = DefaultTransientDetector()
        error = BackendRequestError("rate limited", status_code=429)
        assert detector.classify(error) == TransientAction.RETRY_SAME

    def test_500_is_retry_next(self) -> None:
        """500 server error should retry next backend."""
        detector = DefaultTransientDetector()
        error = BackendRequestError("server error", status_code=500)
        assert detector.classify(error) == TransientAction.RETRY_NEXT

    def test_503_is_retry_next(self) -> None:
        """503 unavailable should retry next backend."""
        detector = DefaultTransientDetector()
        error = BackendRequestError("unavailable", status_code=503)
        assert detector.classify(error) == TransientAction.RETRY_NEXT

    def test_400_is_fail(self) -> None:
        """400 client error should fail immediately."""
        detector = DefaultTransientDetector()
        error = BackendRequestError("bad request", status_code=400)
        assert detector.classify(error) == TransientAction.FAIL

    def test_401_is_fail(self) -> None:
        """401 unauthorized should fail immediately."""
        detector = DefaultTransientDetector()
        error = BackendRequestError("unauthorized", status_code=401)
        assert detector.classify(error) == TransientAction.FAIL


class TestDefaultStrategy:
    """Tests for DefaultStrategy."""

    def test_select_with_explicit_backend(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should return decision with explicit backend."""
        strategy = DefaultStrategy()
        context = RoutingContext(request=mock_request, backend="primary")
        decision = strategy.select(mock_router, context)
        assert decision is not None
        assert decision.backend == "primary"

    def test_select_without_backend_returns_none(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should return None when no backend specified."""
        strategy = DefaultStrategy()
        context = RoutingContext(request=mock_request)
        decision = strategy.select(mock_router, context)
        assert decision is None

    def test_on_error_returns_none(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Default strategy should not retry on error."""
        strategy = DefaultStrategy()
        context = RoutingContext(request=mock_request)
        decision = RoutingDecision(backend="primary")
        result = RoutingResult(
            backend="primary",
            model=None,
            context=context,
            decision=decision,
            error=BackendRequestError("error", status_code=500),
        )
        assert strategy.on_error(mock_router, result) is None

    def test_on_result_is_noop(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """on_result should do nothing."""
        strategy = DefaultStrategy()
        context = RoutingContext(request=mock_request)
        decision = RoutingDecision(backend="primary")
        result = RoutingResult(
            backend="primary",
            model="gpt-4",
            context=context,
            decision=decision,
        )
        strategy.on_result(mock_router, result)  # Should not raise


class TestFallbackStrategy:
    """Tests for FallbackStrategy."""

    def test_select_returns_first_backend(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should return first backend from order."""
        strategy = FallbackStrategy(order=["primary", "fallback"])
        context = RoutingContext(request=mock_request)
        decision = strategy.select(mock_router, context)
        assert decision is not None
        assert decision.backend == "primary"

    def test_select_with_explicit_backend_skips_fallback(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Explicit backend should not use fallback order."""
        strategy = FallbackStrategy(order=["primary", "fallback"])
        context = RoutingContext(request=mock_request, backend="fallback")
        decision = strategy.select(mock_router, context)
        assert decision is not None
        assert decision.backend == "fallback"

    def test_select_with_role_uses_role_order(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should use role-specific order when role matches."""
        strategy = FallbackStrategy(
            order=["primary", "fallback"],
            roles={"special": ["fallback", "primary"]},
        )
        context = RoutingContext(request=mock_request, role="special")
        decision = strategy.select(mock_router, context)
        assert decision is not None
        assert decision.backend == "fallback"

    def test_select_skips_unavailable_backends(self, mock_request: ChatRequest) -> None:
        """Should skip backends not in router.clients."""
        router = MagicMock()
        router.clients = {"fallback": MagicMock()}  # primary not available
        strategy = FallbackStrategy(order=["primary", "fallback"])
        context = RoutingContext(request=mock_request)
        decision = strategy.select(router, context)
        assert decision is not None
        assert decision.backend == "fallback"

    def test_on_error_retries_next_on_5xx(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should try next backend on 5xx error."""
        strategy = FallbackStrategy(order=["primary", "fallback"])
        context = RoutingContext(request=mock_request)
        context.metadata._fallback_order = ["primary", "fallback"]
        context.metadata._fallback_tried = ["primary"]
        decision = RoutingDecision(backend="primary")
        result = RoutingResult(
            backend="primary",
            model=None,
            context=context,
            decision=decision,
            error=BackendRequestError("server error", status_code=500),
        )
        next_decision = strategy.on_error(mock_router, result)
        assert next_decision is not None
        assert next_decision.backend == "fallback"

    def test_on_error_retries_same_on_429(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should retry same backend on 429."""
        strategy = FallbackStrategy(order=["primary", "fallback"])
        context = RoutingContext(request=mock_request)
        decision = RoutingDecision(backend="primary")
        result = RoutingResult(
            backend="primary",
            model=None,
            context=context,
            decision=decision,
            error=BackendRequestError("rate limited", status_code=429),
        )
        next_decision = strategy.on_error(mock_router, result)
        assert next_decision is not None
        assert next_decision.backend == "primary"

    def test_on_error_respects_max_same_retries(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should stop retrying same backend after max attempts."""
        strategy = FallbackStrategy(order=["primary"], max_same_retries=2)
        context = RoutingContext(request=mock_request)
        context.metadata._same_retry_count = 2  # Already at max
        decision = RoutingDecision(backend="primary")
        result = RoutingResult(
            backend="primary",
            model=None,
            context=context,
            decision=decision,
            error=BackendRequestError("rate limited", status_code=429),
        )
        next_decision = strategy.on_error(mock_router, result)
        assert next_decision is None

    def test_on_error_returns_none_on_fatal_error(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should not retry on fatal (4xx) errors."""
        strategy = FallbackStrategy(order=["primary", "fallback"])
        context = RoutingContext(request=mock_request)
        decision = RoutingDecision(backend="primary")
        result = RoutingResult(
            backend="primary",
            model=None,
            context=context,
            decision=decision,
            error=BackendRequestError("bad request", status_code=400),
        )
        next_decision = strategy.on_error(mock_router, result)
        assert next_decision is None

    def test_on_error_no_fallback_for_pinned_backend(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should not fallback to other backends when backend is pinned."""
        strategy = FallbackStrategy(order=["primary", "fallback"])
        context = RoutingContext(request=mock_request, backend="primary")
        decision = RoutingDecision(backend="primary")
        result = RoutingResult(
            backend="primary",
            model=None,
            context=context,
            decision=decision,
            error=BackendRequestError("server error", status_code=500),
        )
        next_decision = strategy.on_error(mock_router, result)
        assert next_decision is None  # No fallback for pinned backend

    def test_on_error_allows_retry_same_for_pinned_backend(
        self, mock_router: MagicMock, mock_request: ChatRequest
    ) -> None:
        """Should allow RETRY_SAME even for pinned backend."""
        strategy = FallbackStrategy(order=["primary", "fallback"])
        context = RoutingContext(request=mock_request, backend="primary")
        decision = RoutingDecision(backend="primary")
        result = RoutingResult(
            backend="primary",
            model=None,
            context=context,
            decision=decision,
            error=BackendRequestError("rate limited", status_code=429),
        )
        next_decision = strategy.on_error(mock_router, result)
        assert next_decision is not None
        assert next_decision.backend == "primary"


class TestDefaultStrategyFactory:
    """Tests for DefaultStrategyFactory."""

    def test_create_returns_default_strategy(self) -> None:
        """Should create DefaultStrategy."""
        from appinfra.dot_dict import DotDict

        factory = DefaultStrategyFactory()
        lg = MagicMock(spec=Logger)
        strategy = factory.create(lg, DotDict())
        assert isinstance(strategy, DefaultStrategy)


class TestFallbackStrategyFactory:
    """Tests for FallbackStrategyFactory."""

    def test_create_with_order(self) -> None:
        """Should create FallbackStrategy with order."""
        from appinfra.dot_dict import DotDict

        factory = FallbackStrategyFactory()
        lg = MagicMock(spec=Logger)
        config = DotDict(order=["a", "b"])
        strategy = factory.create(lg, config)
        assert isinstance(strategy, FallbackStrategy)
        assert strategy._order == ["a", "b"]

    def test_create_with_roles(self) -> None:
        """Should create FallbackStrategy with roles."""
        from appinfra.dot_dict import DotDict

        factory = FallbackStrategyFactory()
        lg = MagicMock(spec=Logger)
        config = DotDict(order=["a"], roles={"special": ["b", "a"]})
        strategy = factory.create(lg, config)
        assert strategy._roles == {"special": ["b", "a"]}

    def test_create_with_max_same_retries(self) -> None:
        """Should create FallbackStrategy with custom max_same_retries."""
        from appinfra.dot_dict import DotDict

        factory = FallbackStrategyFactory()
        lg = MagicMock(spec=Logger)
        config = DotDict(order=["a"], max_same_retries=5)
        strategy = factory.create(lg, config)
        assert strategy._max_same_retries == 5
