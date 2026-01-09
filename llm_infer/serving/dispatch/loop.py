"""Engine loop for processing requests from queue."""

from __future__ import annotations

import multiprocessing as mp
import threading
from queue import Empty
from typing import TYPE_CHECKING

from .handler import RequestHandler
from .types import MetricsRequest, MetricsResponse, RequestStatus, Response

if TYPE_CHECKING:
    from appinfra.log import Logger


def _process_incoming_request(
    handler: RequestHandler, request, response_q: mp.Queue
) -> None:
    """Process a single incoming request from the queue."""
    if isinstance(request, MetricsRequest):
        _handle_metrics_request(request, handler, response_q)
        return
    if not handler.submit(request):
        response_q.put(
            Response(
                id=request.id, status=RequestStatus.REJECTED, error="Server at capacity"
            )
        )


def run_engine_loop(
    handler: RequestHandler,
    request_q: mp.Queue,
    response_q: mp.Queue,
    shutdown: threading.Event,
    lg: Logger | None = None,
    poll_timeout: float = 0.01,
) -> None:
    """Main loop: read requests, process via handler, send responses."""
    handler.set_response_queue(response_q)
    if lg:
        handler.set_logger(lg)

    while not shutdown.is_set():
        try:
            _process_incoming_request(
                handler, request_q.get(timeout=poll_timeout), response_q
            )
        except Empty:
            pass
        for response in handler.step():
            response_q.put(response)


def _handle_metrics_request(
    request: MetricsRequest,
    handler: RequestHandler,
    response_q: mp.Queue,
) -> None:
    """Handle a metrics request by collecting stats and sending response."""
    engine = handler.engine
    stats = engine.memory_stats()
    seq_stats = handler.sequence_stats()

    if request.reset_peak:
        engine.reset_peak_memory()

    response_q.put(
        MetricsResponse(
            id=request.id,
            gpu_allocated_bytes=stats["allocated"],
            gpu_reserved_bytes=stats["reserved"],
            gpu_peak_bytes=stats["peak"],
            kv_cache_bytes=stats["kv_cache_bytes"],
            kv_blocks_used=stats["kv_blocks_used"],
            kv_blocks_total=stats["kv_blocks_total"],
            kv_block_size=stats["kv_block_size"],
            active_sequences=seq_stats["active"],
            total_sequence_tokens=seq_stats["total_tokens"],
            pending_requests=handler.pending_count,
        )
    )


def run_engine_loop_async(
    handler: RequestHandler,
    request_q: mp.Queue,
    response_q: mp.Queue,
    shutdown: threading.Event,
) -> threading.Thread:
    """
    Start the engine loop in a background thread.

    Args:
        handler: The request execution strategy.
        request_q: Queue to receive requests from uvicorn.
        response_q: Queue to send responses to uvicorn.
        shutdown: Event to signal shutdown.

    Returns:
        The background thread running the loop.
    """
    thread = threading.Thread(
        target=run_engine_loop,
        args=(handler, request_q, response_q, shutdown),
        daemon=True,
    )
    thread.start()
    return thread
