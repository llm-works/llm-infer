"""Sequential request handler - one request at a time."""

from collections import deque
from typing import TYPE_CHECKING, Any

from ....context import Event, RequestContext
from ..handler import RequestHandler
from ..types import Request, RequestStatus, Response, StreamChunk

if TYPE_CHECKING:
    from ....primitives.protocols import InferenceEngineProtocol


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

    def _create_context(self, request: Request) -> RequestContext | None:
        """Create RequestContext for a request if logger is available."""
        if self._lg is None:
            return None
        return RequestContext(id=request.id, lg=self._lg)

    def _process_request(self, request: Request) -> Response:
        """Process a single request and return response."""
        # Create context and attach to request
        ctx = self._create_context(request)
        request.context = ctx
        if ctx:
            ctx.mark(
                Event.REQUESTED, stream=request.stream, max_tokens=request.max_tokens
            )

        if request.stream and self._response_q is not None:
            return self._process_streaming_request(request)
        return self._process_blocking_request(request)

    def _build_generate_params(self, request: Request) -> dict:
        """Build parameters for engine.generate from request."""
        return {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "use_chat_template": request.use_chat_template,
            "stop_sequences": request.stop_sequences,
            "context": request.context,
            "messages": request.messages,
        }

    def _process_blocking_request(self, request: Request) -> Response:
        """Process request with blocking generation (non-streaming)."""
        ctx = request.context
        try:
            result = self.engine.generate(**self._build_generate_params(request))
            if ctx:
                ctx.mark(Event.DECODED)
            prompt_tokens = self.engine.count_tokens(request.prompt)
            completion_tokens = self.engine.count_tokens(result)
            if ctx:
                ctx.mark(
                    Event.COMPLETE,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            return Response(
                id=request.id,
                status=RequestStatus.COMPLETED,
                result=result,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            return Response(id=request.id, status=RequestStatus.FAILED, error=str(e))

    def _stream_tokens_to_queue(self, request: Request, stream: Any) -> None:
        """Stream tokens from generator to response queue."""
        assert self._response_q is not None
        for token in stream:
            chunk = StreamChunk(id=request.id, token=token)
            self._response_q.put(chunk)

    def _send_stream_final_chunk(self, request: Request, stream: Any) -> None:
        """Send final chunk with metadata after streaming completes."""
        assert self._response_q is not None
        final_chunk = StreamChunk(
            id=request.id,
            token="",
            is_final=True,
            finish_reason=stream.finish_reason,
            prompt_tokens=stream.prompt_tokens,
            completion_tokens=stream.completion_tokens,
        )
        self._response_q.put(final_chunk)

    def _build_stream_params(self, request: Request) -> dict:
        """Build parameters for generate_stream_sync from request."""
        return {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "use_chat_template": request.use_chat_template,
            "stop_sequences": request.stop_sequences,
            "context": request.context,
            "messages": request.messages,
        }

    def _finalize_stream(self, request: Request, stream: Any) -> Response:
        """Finalize streaming: send final chunk, mark context, return response."""
        ctx = request.context
        if ctx:
            ctx.mark(Event.DECODED)
        self._send_stream_final_chunk(request, stream)
        if ctx:
            ctx.mark(
                Event.COMPLETE,
                prompt_tokens=stream.prompt_tokens,
                completion_tokens=stream.completion_tokens,
            )
        return Response(
            id=request.id,
            status=RequestStatus.COMPLETED,
            prompt_tokens=stream.prompt_tokens,
            completion_tokens=stream.completion_tokens,
        )

    def _process_streaming_request(self, request: Request) -> Response:
        """Process request with streaming generation."""
        try:
            stream = self.engine.generate_stream_sync(
                **self._build_stream_params(request)
            )
            self._stream_tokens_to_queue(request, stream)
            return self._finalize_stream(request, stream)
        except Exception as e:
            if self._response_q is not None:
                error_chunk = StreamChunk(
                    id=request.id, token="", is_final=True, finish_reason="error"
                )
                self._response_q.put(error_chunk)
            return Response(id=request.id, status=RequestStatus.FAILED, error=str(e))

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
