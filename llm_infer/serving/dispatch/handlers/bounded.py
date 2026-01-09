"""Bounded queue request handler - reject when at capacity."""

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ....context import Event, RequestContext
from ..handler import RequestHandler
from ..types import Request, RequestStatus, Response, StreamChunk

if TYPE_CHECKING:
    from ....pipelines.scheduler import Request as EngineRequest
    from ....primitives.protocols import InferenceEngineProtocol


@dataclass
class RunningRequest:
    """Tracks a request in the batched decode loop."""

    request: Request  # Original HTTP request
    engine_request: "EngineRequest"  # Engine request with KV cache
    output_tokens: list[int] = field(default_factory=list)
    last_streamed_idx: int = 0  # Track tokens already sent to stream


class BoundedQueueHandler(RequestHandler):
    """
    Accept up to max_pending requests, reject beyond.

    Properties:
    - Bounded memory usage
    - Explicit backpressure (rejects when full)
    - Fair FIFO queuing
    - Supports batched decode when max_batch_size > 1
    - Supports streaming via response queue (single-request mode only)

    Use for: production with latency SLOs, memory-constrained environments.
    """

    def __init__(
        self,
        engine: "InferenceEngineProtocol",
        max_pending: int = 10,
        max_batch_size: int = 1,
        batch_streaming: bool = False,
    ):
        """
        Initialize the handler.

        Args:
            engine: The inference engine to use for generation.
            max_pending: Maximum number of pending requests before rejection.
            max_batch_size: Maximum requests to batch in decode phase.
                When > 1, enables batched decode for non-streaming requests.
            batch_streaming: Allow streaming requests to join batched decode.
                When True, streaming requests batch with others for better throughput.
        """
        self._engine = engine
        self.max_pending = max_pending
        self.max_batch_size = max_batch_size
        self.batch_streaming = batch_streaming
        self.queue: deque[Request] = deque()
        self.current: Request | None = None
        # For batched mode: track in-flight requests
        self.running: dict[str, RunningRequest] = {}

    def submit(self, request: Request) -> bool:
        """
        Submit a request for processing.

        Rejects if at capacity.

        Args:
            request: The inference request.

        Returns:
            True if accepted, False if rejected (at capacity).
        """
        if self.pending_count >= self.max_pending:
            return False
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
        """Process requests from the queue."""
        # Use batched mode for non-streaming when batch_size > 1
        if self.max_batch_size > 1:
            return self._step_batched()
        return self._step_single()

    def _step_single(self) -> list[Response]:
        """Process one request at a time (original behavior)."""
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

    def _promote_queue_to_running(self) -> list[Response] | None:
        """Promote queued requests to running. Returns early response if streaming handled."""
        while self.queue and len(self.running) < self.max_batch_size:
            req = self.queue[0]

            # Streaming requests use single-request mode unless batch_streaming enabled
            if req.stream and not self.batch_streaming:
                if not self.running:
                    self.queue.popleft()
                    response = self._process_request(req)
                    return [] if req.stream else [response]
                break

            self.queue.popleft()
            running = self._start_request(req)
            if running is not None:
                self.running[req.id] = running
        return None

    def _stream_new_tokens(self) -> None:
        """Stream new tokens for streaming requests in the batch."""
        if self._response_q is None:
            return
        for req_id, running in self.running.items():
            if not running.request.stream:
                continue
            new_tokens = running.engine_request.output_tokens[
                running.last_streamed_idx :
            ]
            for token_id in new_tokens:
                token_text = self.engine.decode_tokens([token_id])
                chunk = StreamChunk(id=req_id, token=token_text)
                self._response_q.put(chunk)
            running.last_streamed_idx = len(running.engine_request.output_tokens)

    def _collect_finished(self) -> list[Response]:
        """Collect responses from finished requests and clean up."""
        responses = []
        finished_ids = []
        for req_id, running in self.running.items():
            if running.engine_request.is_finished:
                if running.request.stream:
                    self._send_final_stream_chunk(running)
                else:
                    responses.append(self._build_response(running))
                finished_ids.append(req_id)
                self.engine.free_request(running.engine_request)

        for req_id in finished_ids:
            del self.running[req_id]
        return responses

    def _step_batched(self) -> list[Response]:
        """Process multiple requests in batched decode mode."""
        early_response = self._promote_queue_to_running()
        if early_response is not None:
            return early_response

        if not self.running:
            return []

        engine_requests = [r.engine_request for r in self.running.values()]
        self.engine.step_decode(engine_requests)

        self._stream_new_tokens()
        return self._collect_finished()

    def _tokenize_request(self, request: Request) -> list[int]:
        """Tokenize request prompt, applying chat template if appropriate."""
        return self.engine.tokenize(request.prompt, request.use_chat_template)

    def _build_stop_token_ids(self, request: Request) -> set[int]:
        """Build set of stop token IDs from EOS token and stop sequences."""
        return self.engine.build_stop_token_ids(request.stop_sequences)

    def _stream_first_token(self, request: Request, engine_request: Any) -> int:
        """Stream first token for streaming requests. Returns last_streamed_idx."""
        if not request.stream or self._response_q is None:
            return 0
        if not engine_request.output_tokens:
            return 0

        first_token = engine_request.output_tokens[0]
        token_text = self.engine.decode_tokens([first_token])
        chunk = StreamChunk(id=request.id, token=token_text)
        self._response_q.put(chunk)
        return 1

    def _create_engine_request(
        self, request: Request, tokens: list[int], ctx: Any
    ) -> Any:
        """Create engine request with sampling params and stop tokens."""
        from ....pipelines.scheduler import Request as EngineRequest

        stop_token_ids = self._build_stop_token_ids(request)
        return EngineRequest.create(
            prompt_tokens=tokens,
            context=ctx,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop_token_ids=stop_token_ids,
        )

    def _run_prefill(self, engine_request: Any) -> None:
        """Run prefill via engine abstraction."""
        self.engine.prefill_request(engine_request)

    def _start_request(self, request: Request) -> RunningRequest | None:
        """Initialize a request for batched processing."""
        ctx = self._create_context(request)
        request.context = ctx
        if ctx:
            ctx.mark(
                Event.REQUESTED, stream=request.stream, max_tokens=request.max_tokens
            )

        try:
            tokens = self._tokenize_request(request)
            if ctx:
                ctx.mark(Event.TOKENIZED, tokens=len(tokens))

            engine_request = self._create_engine_request(request, tokens, ctx)
            self._run_prefill(engine_request)
            last_streamed_idx = self._stream_first_token(request, engine_request)

            return RunningRequest(
                request=request,
                engine_request=engine_request,
                last_streamed_idx=last_streamed_idx,
            )
        except Exception:
            return None

    def _build_response(self, running: RunningRequest) -> Response:
        """Build response from completed running request."""
        ctx = running.engine_request.context
        if ctx:
            ctx.mark(Event.DECODED)

        output_text = self.engine.decode_tokens(running.engine_request.output_tokens)
        prompt_tokens = len(running.engine_request.prompt_tokens)
        completion_tokens = len(running.engine_request.output_tokens)

        if ctx:
            ctx.mark(
                Event.COMPLETE,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return Response(
            id=running.request.id,
            status=RequestStatus.COMPLETED,
            result=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _get_stream_finish_reason(self, engine_req: Any) -> str:
        """Determine finish reason for streaming request."""
        finish_reason = engine_req.finish_reason or "length"
        if (
            engine_req.output_tokens
            and engine_req.output_tokens[-1] in engine_req.stop_token_ids
        ):
            return "stop"
        return finish_reason

    def _send_final_stream_chunk(self, running: RunningRequest) -> None:
        """Send final StreamChunk for a completed streaming request."""
        ctx = running.engine_request.context
        if ctx:
            ctx.mark(Event.DECODED)

        prompt_tokens = len(running.engine_request.prompt_tokens)
        completion_tokens = len(running.engine_request.output_tokens)

        final_chunk = StreamChunk(
            id=running.request.id,
            token="",
            is_final=True,
            finish_reason=self._get_stream_finish_reason(running.engine_request),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        assert self._response_q is not None
        self._response_q.put(final_chunk)

        if ctx:
            ctx.mark(
                Event.COMPLETE,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    @property
    def pending_count(self) -> int:
        """Number of requests waiting or in progress."""
        count = len(self.queue) + len(self.running)
        if self.current:
            count += 1
        return count

    @property
    def is_saturated(self) -> bool:
        """True if at maximum pending requests."""
        return self.pending_count >= self.max_pending

    def sequence_stats(self) -> dict[str, int]:
        """Return active sequence statistics.

        Returns:
            Dict with 'active' (number of in-flight sequences) and
            'total_tokens' (sum of prompt + output tokens across sequences).
        """
        active = len(self.running)
        total_tokens = sum(
            len(r.engine_request.prompt_tokens) + len(r.engine_request.output_tokens)
            for r in self.running.values()
        )
        return {"active": active, "total_tokens": total_tokens}

    @property
    def engine(self) -> "InferenceEngineProtocol":
        """The inference engine used by this handler."""
        return self._engine
