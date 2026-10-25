# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Engine loop for processing requests from queue."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from queue import Empty
from typing import Any, TypeVar

from appinfra.log import Logger

from .handler import RequestHandler
from .processors import RequestProcessor, create_request_processor_chain

T = TypeVar("T")


def _process_incoming_request(
    handler: RequestHandler,
    request: Any,
    response_q: mp.Queue,  # type: ignore[type-arg]
    processor_chain: RequestProcessor,
) -> None:
    """Process a single incoming request using the processor chain."""
    processor_chain.process(request, handler, response_q)


class LoopTasks:
    """Run callables on the engine loop thread.

    The handler is single-threaded: the loop thread owns it. Code on other
    threads (the admin listener) hands its handler calls over through this.
    """

    def __init__(self) -> None:
        self._tasks: queue.SimpleQueue[tuple[Callable[[], Any], Future[Any]]] = (
            queue.SimpleQueue()
        )

    def call(self, fn: Callable[[], T], timeout: float) -> T:
        """Run fn on the loop thread; return its result, or raise its error.

        Raises TimeoutError if the loop doesn't run it within timeout.
        """
        future: Future[T] = Future()
        self._tasks.put((fn, future))
        return future.result(timeout)

    def run_pending(self) -> None:
        """Run the queued callables (loop thread only)."""
        while True:
            try:
                fn, future = self._tasks.get_nowait()
            except Empty:
                return
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as e:
                future.set_exception(e)


def run_engine_loop(
    lg: Logger,
    handler: RequestHandler,
    request_q: mp.Queue,  # type: ignore[type-arg]
    response_q: mp.Queue,  # type: ignore[type-arg]
    shutdown: threading.Event,
    loop_tasks: LoopTasks,
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
        loop_tasks.run_pending()
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
    loop_tasks: LoopTasks,
) -> threading.Thread:
    """Start the engine loop in a background thread.

    Args:
        lg: Logger instance.
        handler: The request execution strategy.
        request_q: Queue to receive requests from uvicorn.
        response_q: Queue to send responses to uvicorn.
        shutdown: Event to signal shutdown.
        loop_tasks: Calls other threads run on the loop thread.

    Returns:
        The background thread running the loop.
    """
    thread = threading.Thread(
        target=run_engine_loop,
        args=(lg, handler, request_q, response_q, shutdown, loop_tasks),
        daemon=True,
    )
    thread.start()
    return thread
