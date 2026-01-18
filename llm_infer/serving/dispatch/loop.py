"""Engine loop for processing requests from queue."""

from __future__ import annotations

import multiprocessing as mp
import threading
from queue import Empty
from typing import Any

from appinfra.log import Logger

from .handler import RequestHandler
from .processors import RequestProcessor, create_request_processor_chain


def _process_incoming_request(
    handler: RequestHandler,
    request: Any,
    response_q: mp.Queue,  # type: ignore[type-arg]
    processor_chain: RequestProcessor,
) -> None:
    """Process a single incoming request using the processor chain."""
    processor_chain.process(request, handler, response_q)


def run_engine_loop(
    lg: Logger,
    handler: RequestHandler,
    request_q: mp.Queue,  # type: ignore[type-arg]
    response_q: mp.Queue,  # type: ignore[type-arg]
    shutdown: threading.Event,
    poll_timeout: float = 0.01,
) -> None:
    """Main loop: read requests, process via handler, send responses."""
    handler.set_response_queue(response_q)
    handler.set_logger(lg)

    processor_chain = create_request_processor_chain()

    while not shutdown.is_set():
        try:
            request = request_q.get(timeout=poll_timeout)
            _process_incoming_request(handler, request, response_q, processor_chain)
        except Empty:
            pass
        for response in handler.step():
            lg.trace("queueing response", extra={"response_id": response.id})
            response_q.put(response)
            lg.trace("response queued", extra={"response_id": response.id})


def run_engine_loop_async(
    lg: Logger,
    handler: RequestHandler,
    request_q: mp.Queue,
    response_q: mp.Queue,
    shutdown: threading.Event,
) -> threading.Thread:
    """Start the engine loop in a background thread.

    Args:
        lg: Logger instance.
        handler: The request execution strategy.
        request_q: Queue to receive requests from uvicorn.
        response_q: Queue to send responses to uvicorn.
        shutdown: Event to signal shutdown.

    Returns:
        The background thread running the loop.
    """
    thread = threading.Thread(
        target=run_engine_loop,
        args=(lg, handler, request_q, response_q, shutdown),
        daemon=True,
    )
    thread.start()
    return thread
