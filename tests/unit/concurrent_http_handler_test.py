# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for ConcurrentHttpHandler."""

import threading
import time
from collections.abc import Generator
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.handlers import ConcurrentHttpHandler
from llm_infer.serving.dispatch.types import (
    Request,
    RequestStatus,
    Response,
    StreamChunk,
)

pytestmark = pytest.mark.unit


def _create_request(req_id: str = "req-1") -> Request:
    """Create a minimal test request."""
    return Request(id=req_id, prompt="test prompt", max_tokens=100)


def _create_mock_engine() -> MagicMock:
    """Create a mock inference engine."""
    engine = MagicMock()
    engine.generate.return_value = "test response"
    engine.count_tokens.return_value = 10
    return engine


class _AbortableEngine:
    """Engine stand-in with a real abort() (a MagicMock doesn't satisfy
    isinstance() against a runtime-checkable protocol on Python 3.12+)."""

    def __init__(self) -> None:
        self.aborted: list[str] = []

    def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


@pytest.fixture
def handler() -> Generator[ConcurrentHttpHandler, None, None]:
    """Create a handler with deterministic cleanup."""
    h = ConcurrentHttpHandler(_create_mock_engine(), max_pending=10, max_concurrent=2)
    yield h
    h.shutdown()


class TestConcurrentHttpHandlerValidation:
    """Tests for __init__ validation."""

    def test_rejects_zero_max_pending(self) -> None:
        """__init__ rejects max_pending <= 0."""
        with pytest.raises(ValueError, match="max_pending must be positive"):
            ConcurrentHttpHandler(_create_mock_engine(), max_pending=0)

    def test_rejects_negative_max_pending(self) -> None:
        """__init__ rejects negative max_pending."""
        with pytest.raises(ValueError, match="max_pending must be positive"):
            ConcurrentHttpHandler(_create_mock_engine(), max_pending=-1)

    def test_rejects_zero_max_concurrent(self) -> None:
        """__init__ rejects max_concurrent <= 0."""
        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            ConcurrentHttpHandler(_create_mock_engine(), max_concurrent=0)

    def test_rejects_negative_max_concurrent(self) -> None:
        """__init__ rejects negative max_concurrent."""
        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            ConcurrentHttpHandler(_create_mock_engine(), max_concurrent=-1)


class TestConcurrentHttpHandlerSubmit:
    """Tests for submit() method."""

    def test_submit_accepts_request(self, handler: ConcurrentHttpHandler) -> None:
        """Submit accepts request when under capacity."""
        request = _create_request()

        result = handler.submit(request)

        assert result is True
        assert handler.pending_count == 1
        assert len(handler.queue) == 1

    def test_submit_rejects_at_capacity(self) -> None:
        """Submit rejects request when at max_pending."""
        handler = ConcurrentHttpHandler(
            _create_mock_engine(), max_pending=2, max_concurrent=1
        )
        try:
            # Fill to capacity
            handler.submit(_create_request("req-1"))
            handler.submit(_create_request("req-2"))

            # Should reject
            result = handler.submit(_create_request("req-3"))

            assert result is False
            assert handler.pending_count == 2
        finally:
            handler.shutdown()

    def test_submit_rejects_after_shutdown(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        """Submit rejects requests after shutdown called."""
        handler._shutdown = True

        result = handler.submit(_create_request())

        assert result is False

    def test_is_saturated_at_max_pending(self) -> None:
        """is_saturated returns True when at max_pending."""
        handler = ConcurrentHttpHandler(_create_mock_engine(), max_pending=2)
        try:
            handler.submit(_create_request("req-1"))
            handler.submit(_create_request("req-2"))

            assert handler.is_saturated is True
        finally:
            handler.shutdown()

    def test_is_saturated_below_max_pending(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        """is_saturated returns False when below max_pending."""
        handler.submit(_create_request())

        assert handler.is_saturated is False


class TestConcurrentHttpHandlerStep:
    """Tests for step() method."""

    def test_step_promotes_queued_to_in_flight(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        """step() promotes queued requests to in_flight."""
        handler.submit(_create_request("req-1"))
        handler.submit(_create_request("req-2"))
        handler.submit(_create_request("req-3"))

        # First step should promote up to max_concurrent
        handler._promote_to_in_flight()

        assert len(handler.queue) == 1
        assert len(handler.in_flight) == 2

    def test_step_respects_max_concurrent(self, handler: ConcurrentHttpHandler) -> None:
        """step() doesn't exceed max_concurrent in-flight."""
        # Submit more than max_concurrent
        for i in range(5):
            handler.submit(_create_request(f"req-{i}"))

        handler._promote_to_in_flight()

        assert len(handler.in_flight) == 2
        assert len(handler.queue) == 3

    def test_step_collects_completed_futures(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        """step() collects responses from completed futures."""
        # Create a completed future manually
        future: Future[Response] = Future()
        response = Response(
            id="req-1", status=RequestStatus.COMPLETED, result="test output"
        )
        future.set_result(response)
        handler.in_flight["req-1"] = future

        # Collect completed
        responses = handler._collect_completed()

        assert len(responses) == 1
        assert responses[0].id == "req-1"
        assert responses[0].status == RequestStatus.COMPLETED
        assert "req-1" not in handler.in_flight

    def test_step_handles_future_exception(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        """step() handles exceptions from futures gracefully."""
        # Create a future that raises exception
        future: Future[Response] = Future()
        future.set_exception(RuntimeError("test error"))
        handler.in_flight["req-1"] = future

        responses = handler._collect_completed()

        assert len(responses) == 1
        assert responses[0].status == RequestStatus.FAILED
        assert "Thread error" in (responses[0].error or "")


class TestConcurrentHttpHandlerShutdown:
    """Tests for shutdown() method."""

    def test_shutdown_answers_queued_requests_unavailable(self) -> None:
        """shutdown() answers queued requests as UNAVAILABLE, without `error`."""
        handler = ConcurrentHttpHandler(
            _create_mock_engine(), max_pending=10, max_concurrent=1
        )
        # Add requests but don't promote them
        handler.queue.append(_create_request("req-1"))
        handler.queue.append(_create_request("req-2"))

        response_q = MagicMock()
        handler._response_q = response_q

        handler.shutdown()

        assert len(handler.queue) == 0
        assert handler._shutdown is True
        sent = [call[0][0] for call in response_q.put.call_args_list]
        assert [r.id for r in sent] == ["req-1", "req-2"]
        for response in sent:
            assert isinstance(response, Response)
            assert response.status == RequestStatus.UNAVAILABLE
            # An `error` would make appinfra's channel raise ChannelError (500)
            assert response.error is None

    def test_shutdown_answers_streaming_request_with_error_chunk(self) -> None:
        """A streaming request gets a final chunk carrying the shutdown error."""
        handler = ConcurrentHttpHandler(_create_mock_engine(), max_concurrent=1)
        request = _create_request("req-s")
        request.stream = True
        handler.queue.append(request)
        response_q = MagicMock()
        handler._response_q = response_q

        handler.shutdown()

        chunk = response_q.put.call_args[0][0]
        assert isinstance(chunk, StreamChunk)
        assert chunk.id == "req-s"
        assert chunk.is_final is True
        assert chunk.error == "Server shutting down"

    def test_shutdown_sets_flag(self, handler: ConcurrentHttpHandler) -> None:
        """shutdown() sets the shutdown flag."""
        handler.shutdown()

        assert handler._shutdown is True

    def test_shutdown_does_not_wait_for_blocked_in_flight_call(self) -> None:
        """shutdown() returns at once while an engine call is still blocked.

        The in-flight request is answered right away; join() reports the call
        as running until it returns.
        """
        started, release = threading.Event(), threading.Event()

        def blocked_generate(**_: object) -> str:
            started.set()
            release.wait(10)
            return "late"

        engine = _create_mock_engine()
        engine.generate.side_effect = blocked_generate
        handler = ConcurrentHttpHandler(engine, max_concurrent=1)
        response_q = MagicMock()
        handler._response_q = response_q
        handler.submit(_create_request("req-1"))
        handler._promote_to_in_flight()
        assert started.wait(5), "engine call never started"

        start = time.monotonic()
        handler.shutdown()
        elapsed = time.monotonic() - start

        try:
            assert elapsed < 1.0
            answer = response_q.put.call_args[0][0]
            assert answer.id == "req-1"
            assert answer.status == RequestStatus.UNAVAILABLE
            assert handler.join(timeout=0.1) is False
        finally:
            release.set()
        assert handler.join(timeout=5.0) is True

    def test_join_without_in_flight_returns_true(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        """join() returns True at once when nothing is running."""
        handler.shutdown()

        assert handler.join(timeout=0.0) is True


class TestConcurrentHttpHandlerCancel:
    """Tests for cancel(): the API side gave up on a request."""

    def test_cancel_queued_request_removes_it(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        handler.queue.append(_create_request("req-1"))
        handler.queue.append(_create_request("req-2"))

        assert handler.cancel("req-1") is True

        assert [r.id for r in handler.queue] == ["req-2"]

    @pytest.mark.parametrize("abortable", [True, False])
    def test_cancel_in_flight_aborts_and_drops_response(self, abortable: bool) -> None:
        engine = _AbortableEngine() if abortable else _create_mock_engine()
        handler = ConcurrentHttpHandler(engine, max_concurrent=1)
        future: Future[Response] = Future()
        handler.in_flight["req-1"] = future
        handler._in_flight_requests["req-1"] = _create_request("req-1")

        assert handler.cancel("req-1") is True

        if abortable:
            assert engine.aborted == ["req-1"]
        future.set_result(
            Response(id="req-1", status=RequestStatus.FAILED, error="aborted")
        )
        assert handler._collect_completed() == []
        assert handler.pending_count == 0  # slot free for the next request
        assert "req-1" not in handler._cancelled

    def test_cancel_unknown_request_returns_false(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        assert handler.cancel("nope") is False

    def test_shutdown_skips_cancelled_in_flight(self) -> None:
        handler = ConcurrentHttpHandler(_create_mock_engine(), max_concurrent=1)
        handler.in_flight["req-1"] = Future()
        handler._in_flight_requests["req-1"] = _create_request("req-1")
        handler.cancel("req-1")
        response_q = MagicMock()
        handler._response_q = response_q

        handler.shutdown()

        response_q.put.assert_not_called()


class TestConcurrentHttpHandlerProperties:
    """Tests for handler properties."""

    def test_pending_count_includes_queue_and_in_flight(
        self, handler: ConcurrentHttpHandler
    ) -> None:
        """pending_count includes both queued and in-flight requests."""
        handler.queue.append(_create_request("req-1"))
        handler.queue.append(_create_request("req-2"))

        future: Future[Response] = Future()
        handler.in_flight["req-3"] = future

        assert handler.pending_count == 3

    def test_engine_property(self) -> None:
        """engine property returns the configured engine."""
        engine = _create_mock_engine()
        handler = ConcurrentHttpHandler(engine)
        try:
            assert handler.engine is engine
        finally:
            handler.shutdown()


class TestConcurrentHttpHandlerProcessing:
    """Tests for request processing."""

    def test_process_request_threadsafe_catches_exceptions(self) -> None:
        """_process_request_threadsafe catches exceptions and returns failed response."""
        engine = _create_mock_engine()
        engine.generate.side_effect = RuntimeError("engine error")
        handler = ConcurrentHttpHandler(engine)
        try:
            request = _create_request()

            response = handler._process_request_threadsafe(request)

            assert response.status == RequestStatus.FAILED
            assert "engine error" in (response.error or "")
        finally:
            handler.shutdown()

    def test_full_request_lifecycle(self, handler: ConcurrentHttpHandler) -> None:
        """Test complete request lifecycle: submit -> step until complete."""
        # Submit request
        request = _create_request("req-1")
        assert handler.submit(request) is True
        assert handler.pending_count == 1

        # Step until we get a response (mock completes fast, may happen in one step)
        responses: list[Response] = []
        for _ in range(10):
            responses = handler.step()
            if responses:
                break
            time.sleep(0.01)

        # Should have one response (completed or failed)
        assert len(responses) == 1
        assert responses[0].id == "req-1"
        assert handler.pending_count == 0
