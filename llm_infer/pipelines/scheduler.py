"""Request scheduling and batch management."""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..primitives.kv_cache import BlockPool, SequenceKVCache

if TYPE_CHECKING:
    from ..context import RequestContext


class RequestState(Enum):
    """Request lifecycle states."""

    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class Request:
    """A single inference request."""

    id: str
    prompt_tokens: list[int]
    context: RequestContext | None = None  # Shared context for logging/timing
    output_tokens: list[int] = field(default_factory=list)
    kv_cache: SequenceKVCache = field(default_factory=SequenceKVCache)
    state: RequestState = RequestState.WAITING
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop_token_ids: set[int] = field(default_factory=set)
    # Guard-related fields
    finish_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        prompt_tokens: list[int],
        context: RequestContext | None = None,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stop_token_ids: set[int] | None = None,
    ) -> Request:
        """Create a new request, using context ID if available."""
        return cls(
            id=context.id if context else str(uuid.uuid4()),
            prompt_tokens=prompt_tokens,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids or set(),
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens processed so far."""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        """Check if request is complete."""
        if self.state == RequestState.FINISHED:
            return True
        if self.finish_reason is not None:
            return True
        if len(self.output_tokens) >= self.max_tokens:
            return True
        if self.output_tokens and self.output_tokens[-1] in self.stop_token_ids:
            return True
        return False

    def mark_finished(self) -> None:
        """Mark the request as finished."""
        self.state = RequestState.FINISHED

    def finish(self, reason: str, message: str | None = None) -> None:
        """Finish the request with a specific reason.

        Args:
            reason: Why generation stopped (e.g., "guard", "max_tokens", "stop_token").
            message: Optional detailed message.
        """
        self.finish_reason = reason
        self.state = RequestState.FINISHED
        if message:
            self.warnings.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message without stopping generation."""
        self.warnings.append(message)


@dataclass
class Scheduler:
    """
    Manages request queue and batch formation.

    For Phase 1, this is a minimal implementation that processes
    one request at a time. Phase 2 will add continuous batching.
    """

    max_batch_size: int = 1  # Phase 1: single request
    waiting: deque[Request] = field(default_factory=deque)
    running: dict[str, Request] = field(default_factory=dict)

    def add_request(self, request: Request) -> str:
        """Add a request to the waiting queue."""
        self.waiting.append(request)
        return request.id

    def get_batch(self) -> tuple[list[Request], list[Request]]:
        """
        Get the next batch of requests to process.

        Returns:
            Tuple of (prefill_requests, decode_requests)
        """
        prefill = []
        decode = []

        # Promote waiting requests to running
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting.popleft()
            request.state = RequestState.PREFILL
            self.running[request.id] = request
            prefill.append(request)

        # Existing running requests continue decoding
        for request in self.running.values():
            if request.state == RequestState.DECODE:
                decode.append(request)

        return prefill, decode

    def update_after_step(self, finished_ids: list[str]) -> None:
        """
        Update state after a generation step.

        Promotes prefill requests to decode, removes finished requests.
        """
        # Transition prefill -> decode
        for request in self.running.values():
            if request.state == RequestState.PREFILL:
                request.state = RequestState.DECODE

        # Remove finished requests
        for request_id in finished_ids:
            if request_id in self.running:
                del self.running[request_id]

    def get_request(self, request_id: str) -> Request | None:
        """Get a request by ID."""
        if request_id in self.running:
            return self.running[request_id]
        for req in self.waiting:
            if req.id == request_id:
                return req
        return None

    def cancel_request(self, request_id: str, block_pool: BlockPool) -> bool:
        """Cancel a request and free its resources."""
        # Check running
        if request_id in self.running:
            request = self.running.pop(request_id)
            request.kv_cache.free_all(block_pool)
            return True

        # Check waiting
        for i, req in enumerate(self.waiting):
            if req.id == request_id:
                del self.waiting[i]
                return True

        return False

    @property
    def num_waiting(self) -> int:
        """Number of requests waiting."""
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        """Number of requests currently running."""
        return len(self.running)

    @property
    def is_empty(self) -> bool:
        """Check if scheduler has no requests."""
        return not self.waiting and not self.running
