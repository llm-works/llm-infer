# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Concurrent HTTP request handler for HTTP-based engines (vLLM server, Ollama)."""

from collections import deque
from concurrent.futures import Future, wait
from typing import TYPE_CHECKING

from ....engines.protocol import AbortableEngineProtocol
from ..handler import RequestHandler
from ..types import Request, RequestStatus, Response, StreamChunk, refusal_message
from .daemon_executor import DaemonThreadExecutor

if TYPE_CHECKING:
    from ....engines.protocol import InferenceEngineProtocol

SHUTTING_DOWN_MESSAGE = "Server shutting down"
RELEASED_MESSAGE = "Backend released"


class ConcurrentHttpHandler(RequestHandler):
    """
    Process multiple requests concurrently using a thread pool.

    Designed for HTTP-based engines (vLLM server, Ollama) where the actual
    inference happens remotely. This handler sends multiple concurrent HTTP
    requests to leverage the backend's continuous batching capability.

    Properties:
    - Concurrent HTTP calls up to max_concurrent limit
    - Bounded memory usage (max_pending queue limit)
    - Thread-safe (httpx.Client supports concurrent use)
    - I/O-bound HTTP calls release GIL efficiently
    - Supports streaming via response queue

    Use for: vLLM server, Ollama, or any HTTP-based inference backend.
    """

    def __init__(
        self,
        engine: "InferenceEngineProtocol",
        max_pending: int = 10,
        max_concurrent: int = 4,
    ):
        """
        Initialize the handler.

        Args:
            engine: The inference engine to use for generation.
            max_pending: Maximum number of pending requests before rejection.
            max_concurrent: Maximum concurrent HTTP requests to the backend.
        """
        if max_pending <= 0:
            raise ValueError(f"max_pending must be positive, got {max_pending}")
        if max_concurrent <= 0:
            raise ValueError(f"max_concurrent must be positive, got {max_concurrent}")

        super().__init__()
        self._engine = engine
        self.max_pending = max_pending
        self.max_concurrent = max_concurrent
        self.queue: deque[Request] = deque()
        self.in_flight: dict[str, Future[Response]] = {}
        # Requests behind in_flight, so shutdown() can answer each in its own form
        self._in_flight_requests: dict[str, Request] = {}
        # In-flight requests whose caller gave up: their responses are dropped
        self._cancelled: set[str] = set()
        # Daemon workers: a call still blocked at shutdown (e.g. on an external
        # vLLM server, which shutdown doesn't stop) must not hold up exit
        self._executor = DaemonThreadExecutor(
            max_workers=max_concurrent, thread_name_prefix="http-handler"
        )
        self._shutdown = False
        # Admin release in progress or done: refuse requests until resume()
        self._suspended = False

    def submit(self, request: Request) -> bool:
        """
        Submit a request for processing.

        Rejects if at capacity.

        Args:
            request: The inference request.

        Returns:
            True if accepted, False if rejected (at capacity).
        """
        if self._shutdown or self._suspended:
            return False
        if self.pending_count >= self.max_pending:
            return False
        self.queue.append(request)
        return True

    def step(self) -> list[Response]:
        """
        Execute one processing step.

        Promotes queued requests to in-flight (up to max_concurrent),
        then collects any completed responses.

        Returns:
            List of completed or failed responses (may be empty).
        """
        self._promote_to_in_flight()
        return self._collect_completed()

    def _promote_to_in_flight(self) -> None:
        """Promote queued requests to in-flight up to max_concurrent limit."""
        while self.queue and len(self.in_flight) < self.max_concurrent:
            request = self.queue.popleft()
            future = self._executor.submit(self._process_request_threadsafe, request)
            self.in_flight[request.id] = future
            self._in_flight_requests[request.id] = request

    def _process_request_threadsafe(self, request: Request) -> Response:
        """Process a request in a worker thread.

        Wraps _process_request to ensure exceptions are captured and
        returned as failed responses rather than crashing the thread.
        """
        try:
            return self._process_request(request)
        except Exception as e:
            return Response(
                id=request.id,
                status=RequestStatus.FAILED,
                error=str(e),
            )

    def _collect_completed(self) -> list[Response]:
        """Collect responses from completed futures without blocking."""
        completed = [rid for rid, future in self.in_flight.items() if future.done()]
        responses: list[Response] = []
        for req_id in completed:
            future = self.in_flight.pop(req_id)
            self._in_flight_requests.pop(req_id, None)
            if req_id in self._cancelled:
                # The caller gave up; nobody reads this response
                self._cancelled.discard(req_id)
                if isinstance(self._engine, AbortableEngineProtocol):
                    self._engine.forget_abort(req_id)
                continue
            if (response := self._response_from(req_id, future)) is not None:
                responses.append(response)
        return responses

    @staticmethod
    def _response_from(req_id: str, future: Future[Response]) -> Response | None:
        """Response to send for a finished future; None if already streamed."""
        try:
            response = future.result()
        except Exception as e:
            # Should not happen since _process_request_threadsafe catches exceptions
            return Response(
                id=req_id, status=RequestStatus.FAILED, error=f"Thread error: {e}"
            )
        # A completed streaming request was answered by its chunks
        if response.status == RequestStatus.COMPLETED and response.result is None:
            return None
        return response

    def cancel(self, request_id: str) -> bool:
        """Drop a queued request, or abort a running one through the engine.

        Running requests need an engine that supports abort; with any other
        engine the call runs to completion and only its response is dropped.
        """
        for request in self.queue:
            if request.id == request_id:
                self.queue.remove(request)
                return True
        if request_id not in self.in_flight:
            return False
        self._cancelled.add(request_id)
        if isinstance(self._engine, AbortableEngineProtocol):
            self._engine.abort(request_id)
        return True

    @property
    def pending_count(self) -> int:
        """Number of requests waiting or in progress."""
        return len(self.queue) + len(self.in_flight)

    @property
    def is_saturated(self) -> bool:
        """True if at maximum pending requests."""
        return self.pending_count >= self.max_pending

    @property
    def engine(self) -> "InferenceEngineProtocol":
        """The inference engine used by this handler."""
        return self._engine

    def shutdown(self) -> None:
        """Answer queued and in-flight requests as unavailable; don't wait.

        An in-flight HTTP call can't be interrupted from this thread: closing
        the httpx client does not wake a read blocked in a worker. The calls end
        when the engine stops an auto-started server; join() waits for them.
        Calls on an external server may outlive this; the daemon workers then
        end with the process.
        """
        self._shutdown = True

        running = [
            r for r in self._in_flight_requests.values() if r.id not in self._cancelled
        ]
        unanswered = [*self.queue, *running]
        self.queue.clear()
        if self._response_q is not None:
            for request in unanswered:
                self._response_q.put(
                    refusal_message(
                        request, RequestStatus.UNAVAILABLE, SHUTTING_DOWN_MESSAGE
                    )
                )

        self._executor.shutdown(cancel_futures=True)

    def join(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for in-flight calls to return."""
        _, not_done = wait(list(self.in_flight.values()), timeout=timeout)
        return not not_done

    def rejection(self, request: Request) -> Response | StreamChunk:
        """Answer for a request submit() refused."""
        if self._suspended:
            return refusal_message(request, RequestStatus.RELEASED, RELEASED_MESSAGE)
        return super().rejection(request)

    @property
    def supports_suspend(self) -> bool:
        return True

    def suspend(self) -> None:
        """Answer queued and running requests as released; refuse new ones.

        Running requests are cancelled (aborted through the engine when it can,
        their late responses dropped), so the backend can stop right after.
        """
        self._suspended = True
        unanswered = [*self.queue]
        self.queue.clear()
        for request_id, request in list(self._in_flight_requests.items()):
            if request_id not in self._cancelled:
                unanswered.append(request)
                self.cancel(request_id)
        if self._response_q is not None:
            for request in unanswered:
                self._response_q.put(
                    refusal_message(request, RequestStatus.RELEASED, RELEASED_MESSAGE)
                )

    def resume(self) -> None:
        """Accept requests again after suspend()."""
        self._suspended = False
