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


class TestDefaultStrategyFactory:
    """Tests for DefaultStrategyFactory."""

    def test_create_returns_default_strategy(self) -> None:
        """Should create DefaultStrategy."""
        from appinfra.dot_dict import DotDict

        factory = DefaultStrategyFactory()
        lg = MagicMock(spec=Logger)
        strategy = factory.create(lg, DotDict())
        assert isinstance(strategy, DefaultStrategy)
