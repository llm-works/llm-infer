"""Concurrent HTTP request handler for HTTP-based engines (vLLM server, Ollama)."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from ..handler import RequestHandler
from ..types import Request, RequestStatus, Response

if TYPE_CHECKING:
    from ....engines.protocol import InferenceEngineProtocol


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
        super().__init__()
        self._engine = engine
        self.max_pending = max_pending
        self.max_concurrent = max_concurrent
        self.queue: deque[Request] = deque()
        self.in_flight: dict[str, Future[Response]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="http-handler"
        )
        self._shutdown = False

    def submit(self, request: Request) -> bool:
        """
        Submit a request for processing.

        Rejects if at capacity.

        Args:
            request: The inference request.

        Returns:
            True if accepted, False if rejected (at capacity).
        """
        if self._shutdown:
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
        responses: list[Response] = []
        completed_ids: list[str] = []

        for req_id, future in self.in_flight.items():
            if future.done():
                completed_ids.append(req_id)
                try:
                    response = future.result()
                    # For streaming requests, response was sent via chunks
                    # Only add non-streaming responses to the list
                    if (
                        response.status != RequestStatus.COMPLETED
                        or response.result is not None
                    ):
                        responses.append(response)
                except Exception as e:
                    # Should not happen since _process_request_threadsafe catches exceptions
                    responses.append(
                        Response(
                            id=req_id,
                            status=RequestStatus.FAILED,
                            error=f"Thread error: {e}",
                        )
                    )

        for req_id in completed_ids:
            del self.in_flight[req_id]

        return responses

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
        """Graceful shutdown: fail pending requests and wait for in-flight to complete."""
        self._shutdown = True

        # Fail queued requests that haven't started
        failed_responses: list[Response] = []
        while self.queue:
            request = self.queue.popleft()
            failed_responses.append(
                Response(
                    id=request.id,
                    status=RequestStatus.FAILED,
                    error="Handler shutting down",
                )
            )

        # Put failed responses on the queue if available
        if self._response_q is not None:
            for response in failed_responses:
                self._response_q.put(response)

        # Shutdown executor, waiting for in-flight requests to complete
        self._executor.shutdown(wait=True, cancel_futures=False)
