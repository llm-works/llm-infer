# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""BackendController reacquire, which runs on a thread of its own."""

from __future__ import annotations

import time

from llm_infer.serving.admin.state import BackendState
from tests.unit.serving.admin._controller import (
    Engine,
    make_controller,
    make_handler,
)
from tests.unit.serving.admin._state import LocalBackendState


def _wait_state(shared: LocalBackendState, want: BackendState) -> None:
    deadline = time.monotonic() + 5
    while shared.get() != want:
        assert time.monotonic() < deadline, f"state stayed {shared.get().label}"
        time.sleep(0.01)


def test_reacquire_loads_then_serves() -> None:
    engine, handler = Engine(), make_handler()
    engine.reacquire_gate.clear()  # hold reacquire to observe LOADING
    controller, shared = make_controller(engine, handler, BackendState.RELEASED)

    assert controller.start_reacquire() == BackendState.LOADING
    assert shared.get() == BackendState.LOADING
    assert controller.start_reacquire() == BackendState.LOADING  # already loading

    engine.reacquire_gate.set()
    _wait_state(shared, BackendState.SERVING)
    handler.resume.assert_called_once()
    assert controller.status() == {"state": "serving", "last_error": None}


def test_reacquire_refreshes_adapters_before_starting_backend() -> None:
    engine = Engine()
    controller, shared = make_controller(engine, make_handler(), BackendState.RELEASED)

    controller.start_reacquire()
    _wait_state(shared, BackendState.SERVING)

    # release() first (idempotent cleanup), then refresh, then reacquire
    assert engine.calls == ["release", "refresh_adapters", "reacquire"]


def test_failed_adapter_refresh_stays_released_with_error() -> None:
    engine, handler = Engine(), make_handler()

    def refresh() -> None:
        raise OSError("adapter directory unreadable")

    controller, shared = make_controller(
        engine, handler, BackendState.RELEASED, refresh
    )

    controller.start_reacquire()
    _wait_state(shared, BackendState.RELEASED)

    # release() runs first (idempotent), then refresh fails
    assert engine.calls == ["release"]
    handler.resume.assert_not_called()
    assert controller.status()["last_error"] == "adapter directory unreadable"


def test_failed_resume_releases_engine_so_retry_works() -> None:
    engine, handler = Engine(), make_handler()
    handler.resume.side_effect = TimeoutError("loop thread blocked")
    controller, shared = make_controller(engine, handler, BackendState.RELEASED)

    controller.start_reacquire()
    _wait_state(shared, BackendState.RELEASED)

    # release() first (idempotent), then started, then released when resume failed
    assert engine.calls == ["release", "refresh_adapters", "reacquire", "release"]
    assert "handler resume failed" in controller.status()["last_error"]


def test_stale_process_cleanup_failure_stays_released() -> None:
    """If cleanup of a stale process fails, stay RELEASED with the error."""
    engine, handler = Engine(), make_handler()
    engine.release_error = OSError("zombie process won't die")
    controller, shared = make_controller(engine, handler, BackendState.RELEASED)

    controller.start_reacquire()
    _wait_state(shared, BackendState.RELEASED)

    assert engine.calls == ["release"]  # Cleanup attempted, failed, nothing else ran
    handler.resume.assert_not_called()
    assert "cleanup of stale process failed" in controller.status()["last_error"]


def test_failed_reacquire_stays_released_with_error() -> None:
    engine, handler = Engine(), make_handler()
    engine.reacquire_error = RuntimeError("vllm serve exited with code 1")
    controller, shared = make_controller(engine, handler, BackendState.RELEASED)

    controller.start_reacquire()
    _wait_state(shared, BackendState.RELEASED)

    handler.resume.assert_not_called()
    assert controller.status()["last_error"] == "vllm serve exited with code 1"


def test_failed_resume_after_reacquire_stays_released_with_error() -> None:
    """If engine.reacquire() succeeds but handler.resume() fails, stay RELEASED."""
    engine, handler = Engine(), make_handler()
    handler.resume.side_effect = TimeoutError("loop thread unresponsive")
    controller, shared = make_controller(engine, handler, BackendState.RELEASED)

    controller.start_reacquire()
    _wait_state(shared, BackendState.RELEASED)

    handler.resume.assert_called_once()
    assert "handler resume failed" in controller.status()["last_error"]
