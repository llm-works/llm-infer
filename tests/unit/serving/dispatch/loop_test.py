"""Unit tests for serving/dispatch/loop.py."""

from __future__ import annotations

import threading
from queue import Queue
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.dispatch.loop import (
    _process_incoming_request,
    run_engine_loop,
    run_engine_loop_async,
)
from llm_infer.serving.dispatch.types import (
    Request,
    RequestStatus,
    Response,
)

pytestmark = pytest.mark.unit


def _make_handler(step_returns: list[list[Response]] | None = None) -> Any:
    """Create a mock handler with submit/step/setters."""
    handler = MagicMock()
    if step_returns is not None:
        handler.step.side_effect = step_returns
    else:
        handler.step.return_value = []
    handler.submit.return_value = True
    return handler


# ---------------------------------------------------------------------------
# _process_incoming_request
# ---------------------------------------------------------------------------


def test_process_incoming_request_calls_chain() -> None:
    handler = MagicMock()
    request = MagicMock()
    response_q = MagicMock()
    chain = MagicMock()

    _process_incoming_request(handler, request, response_q, chain)
    chain.process.assert_called_once_with(request, handler, response_q)


# ---------------------------------------------------------------------------
# run_engine_loop
# ---------------------------------------------------------------------------


class TestRunEngineLoop:
    def test_shutdown_immediately(self) -> None:
        """Loop exits immediately if shutdown is already set."""
        lg = MagicMock(spec=Logger)
        handler = _make_handler()
        request_q: Queue = Queue()
        response_q: Queue = Queue()
        shutdown = threading.Event()
        shutdown.set()

        run_engine_loop(lg, handler, request_q, response_q, shutdown)
        # Setters were called once
        handler.set_response_queue.assert_called_once_with(response_q)
        handler.set_logger.assert_called_once_with(lg)

    def test_processes_request_then_shuts_down(self) -> None:
        """Submit a request, then shut down."""
        lg = MagicMock(spec=Logger)
        handler = _make_handler()
        request_q: Queue = Queue()
        response_q: Queue = Queue()
        shutdown = threading.Event()

        request = Request(id="r1", prompt="hi")
        request_q.put(request)

        # Set up handler.step() to also signal shutdown after first call
        call_count = [0]

        def step_side_effect() -> list[Response]:
            call_count[0] += 1
            if call_count[0] >= 1:
                shutdown.set()
            return []

        handler.step.side_effect = step_side_effect

        run_engine_loop(
            lg, handler, request_q, response_q, shutdown, poll_timeout=0.001
        )
        # Submit was called by InferenceProcessor (request flowed through chain)
        handler.submit.assert_called()

    def test_emits_responses_from_handler_step(self) -> None:
        """Responses returned by handler.step() are queued for delivery."""
        lg = MagicMock(spec=Logger)
        handler = MagicMock()
        request_q: Queue = Queue()
        response_q: Queue = Queue()
        shutdown = threading.Event()

        # First step returns a response, then signal shutdown
        first_response = Response(id="r1", status=RequestStatus.COMPLETED, result="ok")
        call_count = [0]

        def step_side_effect() -> list[Response]:
            call_count[0] += 1
            if call_count[0] == 1:
                return [first_response]
            shutdown.set()
            return []

        handler.step.side_effect = step_side_effect

        run_engine_loop(
            lg, handler, request_q, response_q, shutdown, poll_timeout=0.001
        )

        # Response was put on queue
        item = response_q.get_nowait()
        assert item is first_response

    def test_empty_queue_does_not_block(self) -> None:
        """Empty queue with poll_timeout doesn't crash."""
        lg = MagicMock(spec=Logger)
        handler = _make_handler()
        request_q: Queue = Queue()
        response_q: Queue = Queue()
        shutdown = threading.Event()

        # Trigger shutdown after a few iterations
        call_count = [0]

        def step_side_effect() -> list[Response]:
            call_count[0] += 1
            if call_count[0] >= 3:
                shutdown.set()
            return []

        handler.step.side_effect = step_side_effect

        run_engine_loop(
            lg, handler, request_q, response_q, shutdown, poll_timeout=0.001
        )
        assert call_count[0] >= 3


# ---------------------------------------------------------------------------
# run_engine_loop_async (background thread)
# ---------------------------------------------------------------------------


def test_run_engine_loop_async_starts_thread() -> None:
    lg = MagicMock(spec=Logger)
    handler = _make_handler()
    request_q: Queue = Queue()
    response_q: Queue = Queue()
    shutdown = threading.Event()

    thread = run_engine_loop_async(lg, handler, request_q, response_q, shutdown)
    assert isinstance(thread, threading.Thread)
    assert thread.is_alive()
    assert thread.daemon is True

    shutdown.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
