"""Abstract request handler interface."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ...context import Event, RequestContext
from ..adapters import validate_adapter_id
from .types import Request, RequestStatus, Response, ResponseAdapterInfo, StreamChunk


class AdapterError(Exception):
    """Raised when adapter resolution fails."""


def _stable_adapter_id(adapter_id: str) -> int:
    """Generate deterministic integer ID from adapter name.

    Uses SHA-256 hash to ensure consistent IDs across process restarts.
    Python's built-in hash() is randomized per-process since Python 3.3.

    Args:
        adapter_id: The adapter name string.

    Returns:
        Positive 31-bit integer suitable for vLLM's lora_int_id.
    """
    return int(hashlib.sha256(adapter_id.encode()).hexdigest(), 16) % (2**31)


if TYPE_CHECKING:
    from appinfra.log import Logger

    from ...engines.protocol import InferenceEngineProtocol
    from ..adapters import AdapterManager


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

    Subclasses must call super().__init__() in their __init__ method.
    """

    def __init__(self) -> None:
        """Initialize base handler state.

        Subclasses must call super().__init__() to ensure proper initialization.
        """
        self._response_q: mp.Queue | None = None
        self._lg: Logger | None = None
        self._lora_base_path: Path | None = None
        self._adapter_manager: AdapterManager | None = None
        self._loaded_adapters: set[str] | None = None

    def set_logger(self, lg: Logger) -> None:
        """Set the logger for request context creation."""
        self._lg = lg

    def set_adapter_manager(self, manager: AdapterManager | None) -> None:
        """Set the adapter manager for validation.

        When set, adapter_id must be registered (enabled) in the manager
        for inference to proceed. This enforces the `enabled` field in
        adapter config.yaml files.

        Args:
            manager: AdapterManager instance for validation, or None to
                skip enabled-check (fall back to path-only validation).
        """
        self._adapter_manager = manager

    def get_adapter_manager(self) -> AdapterManager | None:
        """Get the adapter manager for external access (e.g., processors)."""
        return self._adapter_manager

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

    def _validate_adapter_path(self, adapter_id: str) -> Path:
        """Validate adapter_id and return resolved path.

        Raises:
            AdapterError: If LoRA not configured or adapter_id is invalid.
        """
        if not self._lora_base_path:
            raise AdapterError(
                f"adapter_id '{adapter_id}' specified but LoRA not configured "
                "(lora.base_path not set)"
            )

        adapter_path = validate_adapter_id(adapter_id, self._lora_base_path)
        if adapter_path is None:
            if self._lg:
                self._lg.warning(
                    "rejected invalid adapter_id", extra={"adapter_id": adapter_id}
                )
            raise AdapterError(
                f"invalid adapter_id '{adapter_id}': must be a simple name without "
                "path separators or parent references"
            )
        return adapter_path

    def _check_adapter_enabled(self, adapter_id: str, adapter_path: Path) -> None:
        """Check adapter is enabled for use.

        If AdapterManager is set, uses its cached enabled-adapters list.
        Otherwise falls back to reading config.yaml from disk (useful for
        testing or when manager is not configured).

        To hot-reload adapters, use the /v1/adapters/refresh endpoint.

        Raises:
            AdapterError: If adapter is not enabled or config is missing/invalid.
        """
        # Use manager if available (preferred - avoids per-request disk I/O)
        if self._adapter_manager is not None:
            if not self._adapter_manager.is_available(adapter_id):
                raise AdapterError(
                    f"adapter '{adapter_id}' is not enabled. "
                    "Enable it in the adapter's config.yaml and call /v1/adapters/refresh."
                )
            return

        # Fallback: read config.yaml directly (for testing or when manager not set)
        self._check_adapter_config_file(adapter_id, adapter_path)

    def _check_adapter_config_file(self, adapter_id: str, adapter_path: Path) -> None:
        """Check adapter config.yaml exists and has enabled: true."""
        config_path = adapter_path / "config.yaml"
        if not config_path.exists():
            raise AdapterError(
                f"adapter '{adapter_id}' has no config.yaml. "
                "Create config.yaml with 'enabled: true'."
            )

        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            raise AdapterError(
                f"failed to read adapter '{adapter_id}' config: {e}"
            ) from e

        if not isinstance(config, dict):
            raise AdapterError(
                f"adapter '{adapter_id}' config.yaml must be a mapping, "
                f"got {type(config).__name__}"
            )

        if not config.get("enabled", False):
            raise AdapterError(
                f"adapter '{adapter_id}' is not enabled. "
                "Set 'enabled: true' in the adapter's config.yaml."
            )

    def _import_lora_request_class(self, adapter_id: str) -> type[Any]:
        """Import and return vLLM LoRARequest class, raising AdapterError if unavailable."""
        try:
            from vllm.lora.request import LoRARequest

            return LoRARequest  # type: ignore[no-any-return]
        except ImportError as e:
            if self._lg:
                self._lg.warning(
                    "vLLM LoRA module not available", extra={"adapter_id": adapter_id}
                )
            raise AdapterError(
                f"adapter_id '{adapter_id}' specified but vLLM LoRA module not available"
            ) from e

    def _log_and_track_adapter(self, adapter_id: str, adapter_path: Path) -> None:
        """Log adapter usage and track as loaded. Warns on first load."""
        if self._loaded_adapters is None:
            self._loaded_adapters = set()
        is_first_load = adapter_id not in self._loaded_adapters
        if self._lg:
            if is_first_load:
                self._lg.info(
                    "loading LoRA adapter (first use, kernel compilation may take 1-2 min)",
                    extra={"adapter_id": adapter_id, "path": str(adapter_path)},
                )
            else:
                self._lg.debug("using LoRA adapter", extra={"adapter_id": adapter_id})
        self._loaded_adapters.add(adapter_id)

    def _resolve_lora_request(
        self, adapter_id: str | None
    ) -> tuple[Any | None, str | None]:
        """Resolve adapter_id to a vLLM LoRARequest.

        Returns:
            Tuple of (lora_request, fallback_adapter_id):
            - (LoRARequest, None) if adapter resolved successfully
            - (None, None) if no adapter requested
            - (None, adapter_id) if adapter not found (fallback to base model)

        Raises:
            AdapterError: If adapter_id is empty string, adapter is disabled,
                or LoRA module unavailable.
        """
        if adapter_id is None:
            return None, None
        if not adapter_id:
            raise AdapterError("adapter_id cannot be empty")

        adapter_path = self._validate_adapter_path(adapter_id)
        if not adapter_path.exists():
            if self._lg:
                self._lg.warning(
                    "adapter not found, falling back to base model",
                    extra={"adapter_id": adapter_id, "path": str(adapter_path)},
                )
            return None, adapter_id  # Fallback to base model
        self._check_adapter_enabled(adapter_id, adapter_path)
        self._log_and_track_adapter(adapter_id, adapter_path)

        lora_request = self._import_lora_request_class(adapter_id)(
            lora_name=adapter_id,
            lora_int_id=_stable_adapter_id(adapter_id),
            lora_path=str(adapter_path),
        )
        return lora_request, None

    def _build_engine_params(
        self, request: Request
    ) -> tuple[dict[str, Any], str | None]:
        """Build parameters for engine generate/stream methods from request.

        Returns:
            Tuple of (params_dict, fallback_adapter_id). fallback_adapter_id is
            set if adapter was requested but not found (fell back to base model).
        """
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
        lora_request, fallback_adapter_id = self._resolve_lora_request(
            request.adapter_id
        )
        if lora_request is not None:
            params["lora_request"] = lora_request
        # Tool calling support
        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice is not None:
            params["tool_choice"] = request.tool_choice
        # Structured output support
        if request.response_format is not None:
            params["response_format"] = request.response_format
        return params, fallback_adapter_id

    def _parse_generate_result(
        self, result: str | dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any] | None, str | None]:
        """Parse engine generate result.

        Returns:
            Tuple of (content, tool_calls, usage, adapter_mismatch_id).
            adapter_mismatch_id is set if vLLM used a different model than requested.
        """
        if isinstance(result, dict):
            # Check for adapter mismatch (vLLM used different model than requested)
            adapter_mismatch_id = None
            if result.get("adapter_mismatch"):
                adapter_mismatch_id = result.get("adapter_requested")
            return (
                result.get("content", ""),
                result.get("tool_calls"),
                result.get("usage"),
                adapter_mismatch_id,
            )
        return result, None, None, None

    def _build_success_response(
        self,
        request: Request,
        result_text: str,
        tool_calls: list[dict[str, Any]] | None,
        usage: dict[str, Any] | None = None,
        adapter_info: ResponseAdapterInfo | None = None,
    ) -> Response:
        """Build successful response with token counts.

        Uses engine-reported usage when available, falls back to estimation.
        """
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = self.engine.count_tokens(request.prompt)
            completion_tokens = self.engine.count_tokens(result_text)
        if request.context:
            request.context.mark(
                Event.COMPLETE,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        return Response(
            id=request.id,
            status=RequestStatus.COMPLETED,
            result=result_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tool_calls,
            adapter=adapter_info,
        )

    def _process_blocking_request(self, request: Request) -> Response:
        """Process request with blocking generation (non-streaming)."""
        try:
            params, fallback_adapter_id = self._build_engine_params(request)
            result = self.engine.generate(**params)
            if request.context:
                request.context.mark(Event.DECODED)
            result_text, tool_calls, usage, adapter_mismatch_id = (
                self._parse_generate_result(result)
            )
            # Adapter mismatch from response takes precedence over pre-request fallback
            effective_fallback = adapter_mismatch_id or fallback_adapter_id
            adapter_info = (
                ResponseAdapterInfo(requested=effective_fallback, fallback=True)
                if effective_fallback
                else None
            )
            return self._build_success_response(
                request, result_text, tool_calls, usage, adapter_info
            )
        except AdapterError as e:
            if self._lg:
                self._lg.warning(
                    "adapter error", extra={"request_id": request.id, "error": str(e)}
                )
            return Response(id=request.id, status=RequestStatus.FAILED, error=str(e))
        except Exception as e:
            return Response(id=request.id, status=RequestStatus.FAILED, error=str(e))

    def _stream_tokens_to_queue(self, request: Request, stream: Any) -> None:
        """Stream tokens from generator to response queue."""
        assert self._response_q is not None
        for token in stream:
            chunk = StreamChunk(id=request.id, token=token)
            self._response_q.put(chunk)

    def _build_adapter_info(
        self,
        fallback_adapter_id: str | None,
        stream: Any | None = None,
    ) -> ResponseAdapterInfo | None:
        """Build AdapterInfo from fallback ID and/or stream mismatch.

        Returns None if no adapter was involved in the request.
        """
        # Check if stream reported adapter mismatch from vLLM response
        if stream is not None:
            stream_mismatch = getattr(stream, "adapter_mismatch", False)
            if stream_mismatch:
                requested = getattr(stream, "adapter_requested", None)
                return ResponseAdapterInfo(requested=requested, fallback=True)

        if fallback_adapter_id is not None:
            return ResponseAdapterInfo(requested=fallback_adapter_id, fallback=True)

        return None

    def _send_stream_final_chunk(
        self,
        request: Request,
        stream: Any,
        fallback_adapter_id: str | None = None,
    ) -> None:
        """Send final chunk with metadata after streaming completes."""
        assert self._response_q is not None
        adapter_info = self._build_adapter_info(fallback_adapter_id, stream)
        final_chunk = StreamChunk(
            id=request.id,
            token="",
            is_final=True,
            finish_reason=stream.finish_reason,
            prompt_tokens=stream.prompt_tokens,
            completion_tokens=stream.completion_tokens,
            tool_calls=getattr(stream, "tool_calls", None),
            adapter=adapter_info,
        )
        self._response_q.put(final_chunk)

    def _finalize_stream(
        self,
        request: Request,
        stream: Any,
        fallback_adapter_id: str | None = None,
    ) -> Response:
        """Finalize streaming: send final chunk, mark context, return response."""
        ctx = request.context
        if ctx:
            ctx.mark(Event.DECODED)
        self._send_stream_final_chunk(request, stream, fallback_adapter_id)
        if ctx:
            ctx.mark(
                Event.COMPLETE,
                prompt_tokens=stream.prompt_tokens,
                completion_tokens=stream.completion_tokens,
            )
        adapter_info = self._build_adapter_info(fallback_adapter_id, stream)
        return Response(
            id=request.id,
            status=RequestStatus.COMPLETED,
            prompt_tokens=stream.prompt_tokens,
            completion_tokens=stream.completion_tokens,
            adapter=adapter_info,
        )

    def _process_streaming_request(self, request: Request) -> Response:
        """Process request with streaming generation."""
        try:
            params, fallback_adapter_id = self._build_engine_params(request)
            stream = self.engine.generate_stream_sync(**params)
            self._stream_tokens_to_queue(request, stream)
            return self._finalize_stream(request, stream, fallback_adapter_id)
        except AdapterError as e:
            # Adapter validation errors (invalid ID, not enabled, not found)
            if self._lg:
                self._lg.warning(
                    "adapter error", extra={"request_id": request.id, "error": str(e)}
                )
            if self._response_q is not None:
                error_chunk = StreamChunk(
                    id=request.id, token="", is_final=True, finish_reason="error"
                )
                self._response_q.put(error_chunk)
            return Response(id=request.id, status=RequestStatus.FAILED, error=str(e))
        except Exception as e:
            if self._response_q is not None:
                error_chunk = StreamChunk(
                    id=request.id, token="", is_final=True, finish_reason="error"
                )
                self._response_q.put(error_chunk)
            return Response(id=request.id, status=RequestStatus.FAILED, error=str(e))
