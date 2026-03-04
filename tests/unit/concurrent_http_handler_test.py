"""Unit tests for ConcurrentHttpHandler."""

import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.handlers import ConcurrentHttpHandler
from llm_infer.serving.dispatch.types import Request, RequestStatus, Response

pytestmark = pytest.mark.unit


def _create_request(req_id: str = "req-1") -> Request:
    """Create a minimal test request."""
    return Request(id=req_id, prompt="test prompt")


def _create_mock_engine() -> MagicMock:
    """Create a mock inference engine."""
    engine = MagicMock()
    engine.generate.return_value = "test response"
    engine.count_tokens.return_value = 10
    return engine


class TestConcurrentHttpHandlerSubmit:
    """Tests for submit() method."""

    def test_submit_accepts_request(self) -> None:
        """Submit accepts request when under capacity."""
        handler = ConcurrentHttpHandler(_create_mock_engine(), max_pending=10)
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

        # Fill to capacity
        handler.submit(_create_request("req-1"))
        handler.submit(_create_request("req-2"))

        # Should reject
        result = handler.submit(_create_request("req-3"))

        assert result is False
        assert handler.pending_count == 2

    def test_submit_rejects_after_shutdown(self) -> None:
        """Submit rejects requests after shutdown called."""
        handler = ConcurrentHttpHandler(_create_mock_engine())
        handler._shutdown = True

        result = handler.submit(_create_request())

        assert result is False

    def test_is_saturated_at_max_pending(self) -> None:
        """is_saturated returns True when at max_pending."""
        handler = ConcurrentHttpHandler(_create_mock_engine(), max_pending=2)
        handler.submit(_create_request("req-1"))
        handler.submit(_create_request("req-2"))

        assert handler.is_saturated is True

    def test_is_saturated_below_max_pending(self) -> None:
        """is_saturated returns False when below max_pending."""
        handler = ConcurrentHttpHandler(_create_mock_engine(), max_pending=10)
        handler.submit(_create_request())

        assert handler.is_saturated is False


class TestConcurrentHttpHandlerStep:
    """Tests for step() method."""

    def test_step_promotes_queued_to_in_flight(self) -> None:
        """step() promotes queued requests to in_flight."""
        handler = ConcurrentHttpHandler(
            _create_mock_engine(), max_pending=10, max_concurrent=2
        )
        handler.submit(_create_request("req-1"))
        handler.submit(_create_request("req-2"))
        handler.submit(_create_request("req-3"))

        # First step should promote up to max_concurrent
        handler._promote_to_in_flight()

        assert len(handler.queue) == 1
        assert len(handler.in_flight) == 2

    def test_step_respects_max_concurrent(self) -> None:
        """step() doesn't exceed max_concurrent in-flight."""
        handler = ConcurrentHttpHandler(
            _create_mock_engine(), max_pending=10, max_concurrent=2
        )
        # Submit more than max_concurrent
        for i in range(5):
            handler.submit(_create_request(f"req-{i}"))

        handler._promote_to_in_flight()

        assert len(handler.in_flight) == 2
        assert len(handler.queue) == 3

    def test_step_collects_completed_futures(self) -> None:
        """step() collects responses from completed futures."""
        engine = _create_mock_engine()
        handler = ConcurrentHttpHandler(engine, max_pending=10, max_concurrent=2)

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

    def test_step_handles_future_exception(self) -> None:
        """step() handles exceptions from futures gracefully."""
        handler = ConcurrentHttpHandler(_create_mock_engine())

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

    def test_shutdown_fails_queued_requests(self) -> None:
        """shutdown() fails queued requests that haven't started."""
        handler = ConcurrentHttpHandler(
            _create_mock_engine(), max_pending=10, max_concurrent=1
        )
        # Add requests but don't promote them
        handler.queue.append(_create_request("req-1"))
        handler.queue.append(_create_request("req-2"))

        # Set up response queue to capture failed responses
        response_q = MagicMock()
        handler._response_q = response_q

        handler.shutdown()

        assert len(handler.queue) == 0
        assert handler._shutdown is True
        # Should have put 2 failed responses on queue
        assert response_q.put.call_count == 2
        for call in response_q.put.call_args_list:
            response = call[0][0]
            assert response.status == RequestStatus.FAILED
            assert "shutting down" in response.error

    def test_shutdown_sets_flag(self) -> None:
        """shutdown() sets the shutdown flag."""
        handler = ConcurrentHttpHandler(_create_mock_engine())

        handler.shutdown()

        assert handler._shutdown is True

    def test_shutdown_waits_for_in_flight(self) -> None:
        """shutdown() waits for in-flight requests to complete."""
        handler = ConcurrentHttpHandler(_create_mock_engine(), max_concurrent=2)

        # Submit and promote a request
        handler.submit(_create_request())
        handler._promote_to_in_flight()

        # shutdown() should complete without error (executor.shutdown waits)
        handler.shutdown()

        assert handler._shutdown is True


class TestConcurrentHttpHandlerProperties:
    """Tests for handler properties."""

    def test_pending_count_includes_queue_and_in_flight(self) -> None:
        """pending_count includes both queued and in-flight requests."""
        handler = ConcurrentHttpHandler(
            _create_mock_engine(), max_pending=10, max_concurrent=2
        )
        handler.queue.append(_create_request("req-1"))
        handler.queue.append(_create_request("req-2"))

        future: Future[Response] = Future()
        handler.in_flight["req-3"] = future

        assert handler.pending_count == 3

    def test_engine_property(self) -> None:
        """engine property returns the configured engine."""
        engine = _create_mock_engine()
        handler = ConcurrentHttpHandler(engine)

        assert handler.engine is engine


class TestConcurrentHttpHandlerProcessing:
    """Tests for request processing."""

    def test_process_request_threadsafe_catches_exceptions(self) -> None:
        """_process_request_threadsafe catches exceptions and returns failed response."""
        engine = _create_mock_engine()
        engine.generate.side_effect = RuntimeError("engine error")
        handler = ConcurrentHttpHandler(engine)
        request = _create_request()

        response = handler._process_request_threadsafe(request)

        assert response.status == RequestStatus.FAILED
        assert "engine error" in (response.error or "")

    def test_full_request_lifecycle(self) -> None:
        """Test complete request lifecycle: submit -> step until complete."""
        engine = _create_mock_engine()
        engine.generate.return_value = "generated text"
        handler = ConcurrentHttpHandler(engine, max_pending=10, max_concurrent=4)

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
