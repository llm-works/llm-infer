# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for BackendController: release and reacquire the backend."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.admin.controller import (
    BackendController,
    ReleaseNotSupportedError,
    StateConflictError,
)
from llm_infer.serving.admin.state import BackendState, SharedBackendState

pytestmark = pytest.mark.unit

T = TypeVar("T")


class _ImmediateLoopTasks:
    """LoopTasks stand-in: runs the call at once, on the calling thread."""

    def call(self, fn: Callable[[], T], timeout: float) -> T:
        return fn()


class _Engine:
    """Releasable engine stand-in; reacquire can be made to fail or to block."""

    def __init__(self, manages_backend: bool = True) -> None:
        self.manages_backend = manages_backend
        self.calls: list[str] = []
        self.reacquire_error: Exception | None = None
        self.reacquire_gate = threading.Event()
        self.reacquire_gate.set()

    def release(self) -> None:
        self.calls.append("release")

    def reacquire(self) -> None:
        self.reacquire_gate.wait(5)
        self.calls.append("reacquire")
        if self.reacquire_error:
            raise self.reacquire_error


def _handler() -> MagicMock:
    handler = MagicMock()
    handler.supports_suspend = True
    return handler


def _controller(
    engine: Any, handler: Any, state: BackendState = BackendState.SERVING
) -> tuple[BackendController, SharedBackendState]:
    shared = SharedBackendState()
    shared.set(state)
    lg = MagicMock(spec=Logger)
    controller = BackendController(lg, engine, handler, shared, _ImmediateLoopTasks())  # type: ignore[arg-type]
    return controller, shared


def _wait_state(shared: SharedBackendState, want: BackendState) -> None:
    deadline = time.monotonic() + 5
    while shared.get() != want:
        assert time.monotonic() < deadline, f"state stayed {shared.get().label}"
        time.sleep(0.01)


def test_release_suspends_handler_then_stops_backend() -> None:
    engine, handler = _Engine(), _handler()
    controller, shared = _controller(engine, handler)

    assert controller.release() == BackendState.RELEASED

    handler.suspend.assert_called_once()
    assert engine.calls == ["release"]
    assert shared.get() == BackendState.RELEASED


def test_release_when_released_is_a_noop() -> None:
    engine = _Engine()
    controller, _ = _controller(engine, _handler(), BackendState.RELEASED)

    assert controller.release() == BackendState.RELEASED
    assert engine.calls == []


@pytest.mark.parametrize(
    "state", [BackendState.LOADING, BackendState.RELEASING, BackendState.INITIALIZING]
)
def test_release_refused_in_transition(state: BackendState) -> None:
    controller, shared = _controller(_Engine(), _handler(), state)

    with pytest.raises(StateConflictError):
        controller.release()
    assert shared.get() == state


@pytest.mark.parametrize(
    ("engine", "handler_suspends"),
    [
        (object(), True),  # engine can't release
        (_Engine(), False),  # handler can't suspend
        (_Engine(manages_backend=False), True),  # external backend
    ],
)
def test_release_not_supported_changes_nothing(
    engine: Any, handler_suspends: bool
) -> None:
    handler = _handler()
    handler.supports_suspend = handler_suspends
    controller, shared = _controller(engine, handler)

    with pytest.raises(ReleaseNotSupportedError):
        controller.release()
    handler.suspend.assert_not_called()
    assert shared.get() == BackendState.SERVING


class _FailingReleaseEngine(_Engine):
    def release(self) -> None:
        raise RuntimeError("stop failed")


def test_failed_release_resumes_serving() -> None:
    engine, handler = _FailingReleaseEngine(), _handler()
    controller, shared = _controller(engine, handler)

    with pytest.raises(RuntimeError):
        controller.release()

    handler.resume.assert_called_once()
    assert shared.get() == BackendState.SERVING


def test_reacquire_loads_then_serves() -> None:
    engine, handler = _Engine(), _handler()
    engine.reacquire_gate.clear()  # hold reacquire to observe LOADING
    controller, shared = _controller(engine, handler, BackendState.RELEASED)

    assert controller.start_reacquire() == BackendState.LOADING
    assert shared.get() == BackendState.LOADING
    assert controller.start_reacquire() == BackendState.LOADING  # already loading

    engine.reacquire_gate.set()
    _wait_state(shared, BackendState.SERVING)
    handler.resume.assert_called_once()
    assert controller.status() == {"state": "serving", "last_error": None}


def test_reacquire_when_serving_is_a_noop() -> None:
    engine = _Engine()
    controller, _ = _controller(engine, _handler())

    assert controller.start_reacquire() == BackendState.SERVING
    assert engine.calls == []


def test_failed_reacquire_stays_released_with_error() -> None:
    engine, handler = _Engine(), _handler()
    engine.reacquire_error = RuntimeError("vllm serve exited with code 1")
    controller, shared = _controller(engine, handler, BackendState.RELEASED)

    controller.start_reacquire()
    _wait_state(shared, BackendState.RELEASED)

    handler.resume.assert_not_called()
    assert controller.status()["last_error"] == "vllm serve exited with code 1"


def test_failed_resume_after_reacquire_stays_released_with_error() -> None:
    """If engine.reacquire() succeeds but handler.resume() fails, stay RELEASED."""
    engine, handler = _Engine(), _handler()
    handler.resume.side_effect = TimeoutError("loop thread unresponsive")
    controller, shared = _controller(engine, handler, BackendState.RELEASED)

    controller.start_reacquire()
    _wait_state(shared, BackendState.RELEASED)

    handler.resume.assert_called_once()
    assert "handler resume failed" in controller.status()["last_error"]


def test_reacquire_refused_while_releasing() -> None:
    controller, _ = _controller(_Engine(), _handler(), BackendState.RELEASING)

    with pytest.raises(StateConflictError):
        controller.start_reacquire()
