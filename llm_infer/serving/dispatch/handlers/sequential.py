"""Sequential request handler - one request at a time."""

from collections import deque
from typing import TYPE_CHECKING

from ..handler import RequestHandler
from ..types import Request, Response

if TYPE_CHECKING:
    from ....engines.protocol import InferenceEngineProtocol


class SequentialHandler(RequestHandler):
    """
    Process one request at a time.

    Properties:
    - Simple, predictable latency
    - Low throughput (no parallelism)
    - No memory pressure from queuing
    - Never rejects (infinite queue)
    - Supports streaming via response queue

    Use for: debugging, development, single-user deployments.
    """

    def __init__(self, engine: "InferenceEngineProtocol"):
        """
        Initialize the handler.

        Args:
            engine: The inference engine to use for generation.
        """
        super().__init__()
        self._engine = engine
        self.queue: deque[Request] = deque()
        self.current: Request | None = None

    def submit(self, request: Request) -> bool:
        """
        Submit a request for processing.

        Always accepts - requests are queued indefinitely.

        Args:
            request: The inference request.

        Returns:
            Always True (never rejects).
        """
        self.queue.append(request)
        return True

    def step(self) -> list[Response]:
        """Process one request from the queue (blocking)."""
        if self.current is None and self.queue:
            self.current = self.queue.popleft()

        if self.current is None:
            return []

        request = self.current
        response = self._process_request(request)
        self.current = None

        # For streaming requests, don't return the response - it was sent via chunks
        if request.stream:
            return []
        return [response]

    @property
    def pending_count(self) -> int:
        """Number of requests waiting or in progress."""
        return len(self.queue) + (1 if self.current else 0)

    @property
    def is_saturated(self) -> bool:
        """Never saturated - always accepts more requests."""
        return False

    @property
    def engine(self) -> "InferenceEngineProtocol":
        """The inference engine used by this handler."""
        return self._engine
