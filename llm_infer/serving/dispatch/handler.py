"""Abstract request handler interface."""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .types import Request, Response

if TYPE_CHECKING:
    from appinfra.log import Logger

    from ...primitives.protocols import InferenceEngineProtocol


class RequestHandler(ABC):
    """
    Abstract base for request execution strategies.

    Implementations define how inference requests are queued, scheduled,
    and executed. The engine loop calls submit() to add requests and
    step() to advance processing.

    For streaming requests, handlers need access to response_q to send
    incremental tokens. Call set_response_queue() before processing.
    """

    _response_q: mp.Queue | None = None
    _lg: Logger | None = None

    def set_logger(self, lg: Logger) -> None:
        """Set the logger for request context creation."""
        self._lg = lg

    def set_response_queue(self, response_q: mp.Queue) -> None:
        """
        Set the response queue for streaming support.

        This allows handlers to put StreamChunk objects directly on the
        queue during generation, enabling token-by-token streaming.

        Args:
            response_q: Queue to send responses/chunks to API layer.
        """
        self._response_q = response_q

    @abstractmethod
    def submit(self, request: Request) -> bool:
        """
        Submit a request for processing.

        Args:
            request: The inference request.

        Returns:
            True if accepted, False if rejected (at capacity).
        """
        pass

    @abstractmethod
    def step(self) -> list[Response]:
        """
        Execute one processing step.

        This is called repeatedly by the engine loop. Each call may:
        - Process one request (Sequential)
        - Process one request from queue (BoundedQueue)
        - Advance all running requests by one token (ContinuousBatching)

        For streaming requests, handlers should put StreamChunk objects
        on self._response_q during processing and only return the final
        Response in the returned list.

        Returns:
            List of completed or rejected responses (may be empty).
        """
        pass

    @property
    @abstractmethod
    def pending_count(self) -> int:
        """Number of requests waiting or in progress."""
        pass

    @property
    @abstractmethod
    def is_saturated(self) -> bool:
        """True if handler cannot accept more requests."""
        pass

    def sequence_stats(self) -> dict[str, int]:
        """Return active sequence statistics.

        Override in subclasses to provide accurate counts.

        Returns:
            Dict with 'active' (number of sequences) and 'total_tokens'.
        """
        return {"active": 0, "total_tokens": 0}

    @property
    @abstractmethod
    def engine(self) -> InferenceEngineProtocol:
        """The inference engine used by this handler."""
        pass
