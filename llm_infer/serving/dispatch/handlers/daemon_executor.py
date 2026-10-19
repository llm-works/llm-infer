# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Fixed-size thread pool whose workers never delay interpreter exit."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")

_WorkItem = tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]


class DaemonThreadExecutor:
    """Run callables on a fixed set of daemon threads.

    concurrent.futures.ThreadPoolExecutor joins its workers at interpreter
    exit, so a worker blocked in a network read holds the process until the
    read times out. These workers are daemon threads: exit doesn't wait for
    them, and the OS closes their sockets, which the peer sees as a disconnect.
    """

    def __init__(self, max_workers: int, thread_name_prefix: str) -> None:
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")
        self._max_workers = max_workers
        self._work: queue.SimpleQueue[_WorkItem | None] = queue.SimpleQueue()
        self._shutdown = False
        self._lock = threading.Lock()
        for i in range(max_workers):
            threading.Thread(
                target=self._worker, name=f"{thread_name_prefix}-{i}", daemon=True
            ).start()

    def submit(self, fn: Callable[..., T], *args: Any) -> Future[T]:
        """Schedule fn(*args); raises RuntimeError after shutdown()."""
        future: Future[T] = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot submit after shutdown")
            self._work.put((future, fn, args))
        return future

    def shutdown(self, cancel_futures: bool = False) -> None:
        """Stop accepting work and let idle workers exit. Never waits.

        cancel_futures cancels work that hasn't started; running calls finish
        on their own (or end with the process).
        """
        with self._lock:
            self._shutdown = True
        if cancel_futures:
            self._cancel_pending()
        for _ in range(self._max_workers):
            self._work.put(None)

    def _cancel_pending(self) -> None:
        while True:
            try:
                item = self._work.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                item[0].cancel()

    def _worker(self) -> None:
        while (item := self._work.get()) is not None:
            future, fn, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as e:
                future.set_exception(e)
