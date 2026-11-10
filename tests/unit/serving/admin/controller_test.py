# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for BackendController: release, and reacquire's state checks.

A reacquire that starts runs on a thread of its own: tests/integration/.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_infer.serving.admin.controller import (
    ReleaseNotSupportedError,
    StateConflictError,
)
from llm_infer.serving.admin.state import BackendState
from tests.support.serving.admin import Engine, make_controller, make_handler


def test_release_suspends_handler_then_stops_backend() -> None:
    engine, handler = Engine(), make_handler()
    controller, shared = make_controller(engine, handler)

    assert controller.release() == BackendState.RELEASED

    handler.suspend.assert_called_once()
    assert engine.calls == ["release"]
    assert shared.get() == BackendState.RELEASED


def test_release_when_released_is_a_noop() -> None:
    engine = Engine()
    controller, _ = make_controller(engine, make_handler(), BackendState.RELEASED)

    assert controller.release() == BackendState.RELEASED
    assert engine.calls == []


@pytest.mark.parametrize(
    "state", [BackendState.LOADING, BackendState.RELEASING, BackendState.INITIALIZING]
)
def test_release_refused_in_transition(state: BackendState) -> None:
    controller, shared = make_controller(Engine(), make_handler(), state)

    with pytest.raises(StateConflictError):
        controller.release()
    assert shared.get() == state


@pytest.mark.parametrize(
    ("engine", "handler_suspends"),
    [
        (object(), True),  # engine can't release
        (Engine(), False),  # handler can't suspend
        (Engine(manages_backend=False), True),  # external backend
    ],
)
def test_release_not_supported_changes_nothing(
    engine: Any, handler_suspends: bool
) -> None:
    handler = make_handler()
    handler.supports_suspend = handler_suspends
    controller, shared = make_controller(engine, handler)

    with pytest.raises(ReleaseNotSupportedError):
        controller.release()
    handler.suspend.assert_not_called()
    assert shared.get() == BackendState.SERVING


class _FailingReleaseEngine(Engine):
    def release(self) -> None:
        raise RuntimeError("stop failed")


def test_failed_release_resumes_serving() -> None:
    engine, handler = _FailingReleaseEngine(), make_handler()
    controller, shared = make_controller(engine, handler)

    with pytest.raises(RuntimeError):
        controller.release()

    handler.resume.assert_called_once()
    assert shared.get() == BackendState.SERVING


def test_reacquire_when_serving_is_a_noop() -> None:
    engine = Engine()
    controller, _ = make_controller(engine, make_handler())

    assert controller.start_reacquire() == BackendState.SERVING
    assert engine.calls == []


def test_reacquire_refused_while_releasing() -> None:
    controller, _ = make_controller(Engine(), make_handler(), BackendState.RELEASING)

    with pytest.raises(StateConflictError):
        controller.start_reacquire()
