"""Abstract request handler interface."""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...context import Event, RequestContext
from .types import Request, RequestStatus, Response, StreamChunk

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

    Template Method: _process_request() defines the request processing
    algorithm, with subclasses implementing submit() and step() for
    queue management.
    """

    _response_q: mp.Queue | None = None
    _lg: Logger | None = None
    _lora_base_path: Path | None = None

    def set_logger(self, lg: Logger) -> None:
        """Set the logger for request context creation."""
        self._lg = lg

    def set_lora_base_path(self, path: str | None) -> None:
        """Set the base path for LoRA adapter resolution.

        Args:
            path: Base directory for adapter weights. Adapter paths are
                constructed as base_path / adapter_id.
        """
        self._lora_base_path = Path(path).expanduser() if path else None

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

    # -------------------------------------------------------------------------
    # Template Method: Request Processing (shared by all handlers)
    # -------------------------------------------------------------------------

    def _create_context(self, request: Request) -> RequestContext | None:
        """Create RequestContext for a request if logger is available."""
        if self._lg is None:
            return None
        return RequestContext(id=request.id, lg=self._lg)

    def _process_request(self, request: Request) -> Response:
        """Process a single request and return response (Template Method)."""
        ctx = self._create_context(request)
        request.context = ctx
        if ctx:
            ctx.mark(
                Event.REQUESTED, stream=request.stream, max_tokens=request.max_tokens
            )

        if request.stream and self._response_q is not None:
            return self._process_streaming_request(request)
        return self._process_blocking_request(request)

    def _resolve_lora_request(self, adapter_id: str | None) -> Any | None:
        """Resolve adapter_id to a vLLM LoRARequest.

        Uses convention-based path resolution: base_path / adapter_id.

        Args:
            adapter_id: Adapter name from the request.

        Returns:
            LoRARequest if adapter_id provided and base_path configured, None otherwise.
        """
        if not adapter_id or not self._lora_base_path:
            return None

        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            return None

        adapter_path = self._lora_base_path / adapter_id
        return LoRARequest(
            lora_name=adapter_id,
            lora_int_id=hash(adapter_id) % (2**31),  # Stable ID from name
            lora_path=str(adapter_path),
        )

    def _build_generate_params(self, request: Request) -> dict[str, Any]:
        """Build parameters for engine.generate from request."""
        params: dict[str, Any] = {
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
        if lora_request := self._resolve_lora_request(request.adapter_id):
            params["lora_request"] = lora_request
        return params

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

    def _build_stream_params(self, request: Request) -> dict[str, Any]:
        """Build parameters for generate_stream_sync from request."""
        params: dict[str, Any] = {
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
        if lora_request := self._resolve_lora_request(request.adapter_id):
            params["lora_request"] = lora_request
        return params

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
