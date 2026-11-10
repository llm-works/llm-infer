# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""BackendController with stand-in engine, handler, state and loop tasks."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar
from unittest.mock import MagicMock

from appinfra.log import Logger

from llm_infer.serving.admin.controller import BackendController
from llm_infer.serving.admin.state import BackendState

from ._state import LocalBackendState

T = TypeVar("T")


class ImmediateLoopTasks:
    """LoopTasks stand-in: runs the call at once, on the calling thread."""

    def call(self, fn: Callable[[], T], timeout: float) -> T:
        return fn()


class Engine:
    """Releasable engine stand-in; reacquire can be made to fail or to block."""

    def __init__(self, manages_backend: bool = True) -> None:
        self.manages_backend = manages_backend
        self.calls: list[str] = []
        self.reacquire_error: Exception | None = None
        self.release_error: Exception | None = None
        self.reacquire_gate = threading.Event()
        self.reacquire_gate.set()

    def release(self) -> None:
        self.calls.append("release")
        if self.release_error:
            raise self.release_error

    def reacquire(self) -> None:
        self.reacquire_gate.wait(5)
        self.calls.append("reacquire")
        if self.reacquire_error:
            raise self.reacquire_error


def make_handler() -> MagicMock:
    handler = MagicMock()
    handler.supports_suspend = True
    return handler


def make_controller(
    engine: Any,
    handler: Any,
    state: BackendState = BackendState.SERVING,
    refresh_adapters: Callable[[], None] | None = None,
) -> tuple[BackendController, LocalBackendState]:
    shared = LocalBackendState()
    shared.set(state)
    lg = MagicMock(spec=Logger)
    if refresh_adapters is None:
        # Records into the engine's calls, so tests see it relative to reacquire
        calls = getattr(engine, "calls", [])
        refresh_adapters = lambda: calls.append("refresh_adapters")  # noqa: E731
    controller = BackendController(
        lg,
        engine,
        handler,
        shared,  # type: ignore[arg-type]
        ImmediateLoopTasks(),  # type: ignore[arg-type]
        refresh_adapters,
    )
    return controller, shared
