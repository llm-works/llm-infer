# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for DaemonThreadExecutor."""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from llm_infer.serving.dispatch.handlers.daemon_executor import DaemonThreadExecutor

pytestmark = pytest.mark.unit


def test_submit_returns_result() -> None:
    executor = DaemonThreadExecutor(2, "t")
    try:
        assert executor.submit(lambda a, b: a + b, 2, 3).result(timeout=5) == 5
    finally:
        executor.shutdown()


def test_exception_is_set_on_future() -> None:
    executor = DaemonThreadExecutor(1, "t")
    try:
        future = executor.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            future.result(timeout=5)
    finally:
        executor.shutdown()


def test_workers_are_daemon_threads() -> None:
    executor = DaemonThreadExecutor(3, "dtest")
    try:
        workers = [t for t in threading.enumerate() if t.name.startswith("dtest-")]
        assert len(workers) == 3
        assert all(t.daemon for t in workers)
    finally:
        executor.shutdown()


def test_rejects_non_positive_max_workers() -> None:
    with pytest.raises(ValueError, match="max_workers must be positive"):
        DaemonThreadExecutor(0, "t")


def test_submit_after_shutdown_raises() -> None:
    executor = DaemonThreadExecutor(1, "t")
    executor.shutdown()

    with pytest.raises(RuntimeError, match="after shutdown"):
        executor.submit(lambda: None)


def test_shutdown_cancels_pending_and_does_not_wait() -> None:
    started, release = threading.Event(), threading.Event()

    def blocked() -> bool:
        started.set()
        return release.wait(10)

    executor = DaemonThreadExecutor(1, "t")
    running = executor.submit(blocked)
    pending = executor.submit(lambda: "never")
    assert started.wait(5)

    start = time.monotonic()
    executor.shutdown(cancel_futures=True)
    elapsed = time.monotonic() - start

    try:
        assert elapsed < 1.0
        assert pending.cancelled()
        assert not running.done()
    finally:
        release.set()
    assert running.result(timeout=5) is True


def test_blocked_worker_does_not_delay_interpreter_exit() -> None:
    """The reason this executor exists: ThreadPoolExecutor would block here."""
    script = (
        "import threading\n"
        "from llm_infer.serving.dispatch.handlers.daemon_executor "
        "import DaemonThreadExecutor\n"
        "ex = DaemonThreadExecutor(1, 't')\n"
        "ex.submit(threading.Event().wait)\n"
        "ex.shutdown(cancel_futures=True)\n"
    )
    start = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", script], timeout=30, check=False)

    assert proc.returncode == 0
    assert time.monotonic() - start < 15.0
