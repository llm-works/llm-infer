"""Request processors using Chain of Responsibility pattern."""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .metrics import build_metrics_response
from .types import (
    AdapterInfo,
    AdapterListRequest,
    AdapterListResponse,
    AdapterRefreshRequest,
    AdapterRefreshResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    MetricsRequest,
    Request,
    RequestStatus,
    Response,
)

if TYPE_CHECKING:
    from .handler import RequestHandler


class RequestProcessor(ABC):
    """Abstract base for request processors in the chain."""

    def __init__(self) -> None:
        self._next: RequestProcessor | None = None

    def set_next(self, processor: RequestProcessor) -> RequestProcessor:
        """Set the next processor in the chain. Returns the next processor for chaining."""
        self._next = processor
        return processor

    def process_next(
        self,
        request: Any,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        """Pass to next processor if exists, or emit error for unhandled requests."""
        if self._next:
            self._next.process(request, handler, response_q)
        else:
            # End of chain with no handler - emit error response
            request_id = getattr(request, "id", "unknown")
            response_q.put(
                Response(
                    id=request_id,
                    status=RequestStatus.FAILED,
                    error=f"Unhandled request type: {type(request).__name__}",
                )
            )

    @abstractmethod
    def can_process(self, request: Any) -> bool:
        """Check if this processor can handle the request."""
        pass

    @abstractmethod
    def handle(
        self,
        request: Any,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        """Handle the request."""
        pass

    def process(
        self,
        request: Any,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        """Process if can handle, otherwise pass to next."""
        if self.can_process(request):
            self.handle(request, handler, response_q)
        else:
            self.process_next(request, handler, response_q)


class MetricsProcessor(RequestProcessor):
    """Handles MetricsRequest by collecting and returning system metrics."""

    def can_process(self, request: Any) -> bool:
        return isinstance(request, MetricsRequest)

    def handle(
        self,
        request: MetricsRequest,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        response = build_metrics_response(request.id, handler, request.reset_peak)
        response_q.put(response)


class AdapterProcessor(RequestProcessor):
    """Handles adapter management requests (list, refresh)."""

    def can_process(self, request: Any) -> bool:
        return isinstance(request, (AdapterListRequest, AdapterRefreshRequest))

    def handle(
        self,
        request: AdapterListRequest | AdapterRefreshRequest,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        if isinstance(request, AdapterListRequest):
            self._handle_list(request, handler, response_q)
        else:
            self._handle_refresh(request, handler, response_q)

    def _handle_list(
        self,
        request: AdapterListRequest,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        """List all loaded adapters."""
        manager = handler._adapter_manager
        if manager is None:
            response_q.put(AdapterListResponse(id=request.id, adapters=[]))
            return

        adapters = [
            AdapterInfo(
                adapter_id=a.adapter_id,
                description=a.description,
                loaded_at=a.loaded_at.isoformat(),
            )
            for a in manager.list()
        ]
        response_q.put(AdapterListResponse(id=request.id, adapters=adapters))

    def _handle_refresh(
        self,
        request: AdapterRefreshRequest,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        """Refresh adapters (single or full scan)."""
        manager = handler._adapter_manager
        if manager is None:
            response_q.put(self._make_refresh_response(request, 0, "disabled"))
            return

        if request.adapter_id:
            adapter = manager.refresh_one(request.adapter_id)
            status = "loaded" if adapter else "unloaded"
            response_q.put(
                self._make_refresh_response(request, len(manager.list()), status)
            )
        else:
            count = manager.scan()
            response_q.put(self._make_refresh_response(request, count, "scanned"))

    def _make_refresh_response(
        self, request: AdapterRefreshRequest, adapters_loaded: int, status: str
    ) -> AdapterRefreshResponse:
        """Create an adapter refresh response."""
        return AdapterRefreshResponse(
            id=request.id,
            adapter_id=request.adapter_id,
            adapters_loaded=adapters_loaded,
            status=status,
        )


def _make_embedding_error(request_id: str, error: str) -> EmbeddingResponse:
    """Create a failed embedding response."""
    return EmbeddingResponse(id=request_id, status=RequestStatus.FAILED, error=error)


def _make_embedding_success(
    request_id: str, embeddings: list[list[float]], total_tokens: int
) -> EmbeddingResponse:
    """Create a successful embedding response."""
    return EmbeddingResponse(
        id=request_id,
        status=RequestStatus.COMPLETED,
        embeddings=embeddings,
        total_tokens=total_tokens,
    )


class EmbeddingProcessor(RequestProcessor):
    """Handles EmbeddingRequest by generating embeddings."""

    def can_process(self, request: Any) -> bool:
        return isinstance(request, EmbeddingRequest)

    def handle(
        self,
        request: EmbeddingRequest,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        engine = handler.engine
        if not getattr(engine, "supports_embeddings", lambda: False)():
            response_q.put(
                _make_embedding_error(request.id, "Engine does not support embeddings")
            )
            return

        try:
            embeddings, total_tokens = engine.embed(request.inputs, request.dimensions)
            response_q.put(
                _make_embedding_success(request.id, embeddings, total_tokens)
            )
        except Exception as e:
            response_q.put(_make_embedding_error(request.id, str(e)))


class InferenceProcessor(RequestProcessor):
    """Handles inference requests by submitting to the handler queue."""

    def can_process(self, request: Any) -> bool:
        # Always handles - this is the default at end of chain
        return isinstance(request, Request)

    def handle(
        self,
        request: Request,
        handler: RequestHandler,
        response_q: mp.Queue[Any],
    ) -> None:
        if not handler.submit(request):
            response_q.put(
                Response(
                    id=request.id,
                    status=RequestStatus.REJECTED,
                    error="Server at capacity",
                )
            )


def create_request_processor_chain() -> RequestProcessor:
    """Create the default request processor chain.

    Chain order:
    1. MetricsProcessor - handles metrics requests
    2. AdapterProcessor - handles adapter management requests
    3. EmbeddingProcessor - handles embedding requests
    4. InferenceProcessor - handles inference requests (default)
    """
    chain = MetricsProcessor()
    chain.set_next(AdapterProcessor()).set_next(EmbeddingProcessor()).set_next(
        InferenceProcessor()
    )
    return chain
