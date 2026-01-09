"""Continuous batching request handler - PLACEHOLDER for future implementation.

Reserved for true continuous batching (batched prefill + batched decode).
Currently falls back to sequential processing.

For batched decode only, use BoundedQueueHandler with max_batch_size > 1.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..handler import RequestHandler
from ..types import Request, RequestStatus, Response

if TYPE_CHECKING:
    from ....primitives.protocols import InferenceEngineProtocol


@dataclass
class RunningRequest:
    """A request currently being processed in a batch."""

    request: Request
    output_tokens: list[int] = field(default_factory=list)
    is_finished: bool = False

    def get_output(self, tokenizer) -> str:
        """Decode output tokens to string."""
        result: str = tokenizer.decode(self.output_tokens, skip_special_tokens=True)
        return result


class ContinuousBatchingHandler(RequestHandler):
    """PLACEHOLDER: Continuous batching handler (not yet implemented).

    This class is reserved for future true continuous batching implementation
    that would batch both prefill and decode phases. Currently falls back to
    sequential processing of requests.

    For production batched decode (prefill sequential, decode batched), use
    BoundedQueueHandler with max_batch_size > 1 instead.

    Full implementation would require:
    - Batched prefill with variable-length sequences
    - Continuous insertion of new requests into running batch
    - Preemption support for long-running requests
    """

    def __init__(
        self,
        engine: "InferenceEngineProtocol",
        max_batch_size: int = 32,
        max_pending: int = 100,
    ):
        """
        Initialize the handler.

        Args:
            engine: The inference engine.
            max_batch_size: Maximum requests to batch together.
            max_pending: Maximum total pending requests before rejection.
        """
        self._engine = engine
        self.max_batch_size = max_batch_size
        self.max_pending = max_pending
        self.waiting: deque[Request] = deque()
        self.running: dict[str, RunningRequest] = {}

    @property
    def engine(self) -> "InferenceEngineProtocol":
        """The inference engine used by this handler."""
        return self._engine

    def submit(self, request: Request) -> bool:
        """
        Submit a request for processing.

        Args:
            request: The inference request.

        Returns:
            True if accepted, False if rejected (at capacity).
        """
        if self.pending_count >= self.max_pending:
            return False
        self.waiting.append(request)
        return True

    def _process_request(self, req_id: str, running: RunningRequest) -> Response:
        """Process a single request and return response."""
        try:
            result = self.engine.generate(
                prompt=running.request.prompt,
                max_tokens=running.request.max_tokens,
                temperature=running.request.temperature,
                top_p=running.request.top_p,
                top_k=running.request.top_k,
                repetition_penalty=running.request.repetition_penalty,
                use_chat_template=running.request.use_chat_template,
                stop_sequences=running.request.stop_sequences,
                messages=running.request.messages,
            )
            prompt_tokens = self.engine.count_tokens(running.request.prompt)
            completion_tokens = self.engine.count_tokens(result)
            return Response(
                id=req_id,
                status=RequestStatus.COMPLETED,
                result=result,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            return Response(id=req_id, status=RequestStatus.FAILED, error=str(e))

    def step(self) -> list[Response]:
        """Execute one batched processing step (stub: sequential fallback)."""
        # Promote waiting -> running
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting.popleft()
            self.running[req.id] = RunningRequest(request=req)

        if not self.running:
            return []

        # STUB: Sequential processing until engine.step_batch() is implemented
        responses = []
        finished_ids = []
        for req_id, running in list(self.running.items()):
            responses.append(self._process_request(req_id, running))
            finished_ids.append(req_id)

        for req_id in finished_ids:
            del self.running[req_id]

        return responses

    @property
    def pending_count(self) -> int:
        """Number of requests waiting or running."""
        return len(self.waiting) + len(self.running)

    @property
    def is_saturated(self) -> bool:
        """True if at maximum pending requests."""
        return self.pending_count >= self.max_pending
