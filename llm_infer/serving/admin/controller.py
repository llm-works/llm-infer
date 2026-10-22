# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Release the inference backend's GPU and reacquire it, while serve keeps running."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from appinfra.log import Logger

from ...engines.protocol import ReleasableEngineProtocol
from ..dispatch.handler import RequestHandler
from ..dispatch.loop import LoopTasks
from .state import BackendState, SharedBackendState

# How long a handler call may wait for the engine loop thread to run it
HANDLER_CALL_TIMEOUT = 10.0


class AdminError(Exception):
    """An admin call that can't be carried out; status_code is its HTTP status."""

    status_code = 409


class ReleaseNotSupportedError(AdminError):
    """The engine or handler can't release, or serve doesn't own the backend."""


class StateConflictError(AdminError):
    """The backend is in a state the call can't start from."""


class BackendController:
    """Stop the backend to free its GPU memory, and start it again.

    release(): new inference is answered 503 (state RELEASING), the handler
    answers what's queued or running and refuses more, then the engine stops the
    backend and returns once its processes are gone (state RELEASED).
    start_reacquire(): state LOADING, refresh_adapters() rescans the adapter
    directory and the engine starts the backend with the result, in a background
    thread, then the handler accepts requests again (state SERVING). Adapters
    deployed or removed while released are served or gone from then on.
    A failed reacquire leaves RELEASED, with the error in status().

    Transitions are checked under a lock; the long steps run outside it, the
    RELEASING / LOADING state keeping other calls out meanwhile.
    """

    def __init__(
        self,
        lg: Logger,
        engine: Any,
        handler: RequestHandler,
        state: SharedBackendState,
        loop_tasks: LoopTasks,
        refresh_adapters: Callable[[], None],
    ) -> None:
        self._lg = lg
        self._engine = engine
        self._handler = handler
        self._state = state
        self._loop_tasks = loop_tasks
        self._refresh_adapters = refresh_adapters
        self._lock = threading.Lock()
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        return {"state": self._state.get().label, "last_error": self._last_error}

    def release(self) -> BackendState:
        """Stop the backend; returns once its GPU memory is freed. Idempotent."""
        with self._lock:
            current = self._state.get()
            if current == BackendState.RELEASED:
                return current
            if current != BackendState.SERVING:
                raise StateConflictError(f"backend is {current.label}")
            self._check_releasable()
            self._state.set(BackendState.RELEASING)
        self._lg.info("releasing backend")
        try:
            self._loop_tasks.call(self._handler.suspend, HANDLER_CALL_TIMEOUT)
            self._engine.release()
        except Exception as e:
            self._lg.error("release failed", extra={"exception": e})
            try:
                self._resume_serving()
            except Exception as resume_err:
                self._lg.error(
                    "resume after failed release also failed",
                    extra={"exception": resume_err},
                )
            raise
        self._state.set(BackendState.RELEASED)
        self._lg.info("backend released")
        return BackendState.RELEASED

    def _check_releasable(self) -> None:
        if not isinstance(self._engine, ReleasableEngineProtocol):
            raise ReleaseNotSupportedError("engine can't release its backend")
        if not self._handler.supports_suspend:
            raise ReleaseNotSupportedError("request handler can't suspend")
        if not self._engine.manages_backend:
            raise ReleaseNotSupportedError("backend is not managed by serve")

    def start_reacquire(self) -> BackendState:
        """Start the backend again in the background; returns the new state.

        SERVING: nothing to do. LOADING: started now, or already loading.
        """
        with self._lock:
            current = self._state.get()
            if current in (BackendState.SERVING, BackendState.LOADING):
                return current
            if current != BackendState.RELEASED:
                raise StateConflictError(f"backend is {current.label}")
            self._state.set(BackendState.LOADING)
            self._last_error = None
        threading.Thread(target=self._reacquire, name="reacquire", daemon=True).start()
        return BackendState.LOADING

    def _reacquire(self) -> None:
        self._lg.info("reacquiring backend")
        try:
            self._refresh_adapters()
            self._engine.reacquire()
        except Exception as e:
            self._last_error = str(e)
            self._lg.error("reacquire failed", extra={"exception": e})
            self._state.set(BackendState.RELEASED)
            return
        try:
            self._resume_serving()
        except Exception as e:
            self._last_error = f"handler resume failed: {e}"
            self._lg.error("handler resume failed", extra={"exception": e})
            self._state.set(BackendState.RELEASED)
            return
        self._lg.info("backend reacquired")

    def _resume_serving(self) -> None:
        """Let the handler accept requests again and mark the backend SERVING."""
        self._loop_tasks.call(self._handler.resume, HANDLER_CALL_TIMEOUT)
        self._state.set(BackendState.SERVING)
