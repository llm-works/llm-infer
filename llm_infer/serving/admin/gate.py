# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Answer inference with 503 while an admin has the backend released."""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .state import BackendState, SharedBackendState

# Routes that need the backend; health, models and metrics keep answering
GATED_PATHS = frozenset(
    {"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/generate"}
)

# Backend states from an admin release until reacquired
_UNAVAILABLE = frozenset(
    {BackendState.RELEASING, BackendState.RELEASED, BackendState.LOADING}
)


class BackendGateMiddleware:
    """ASGI middleware for the HTTP subprocess: 503 with Retry-After on inference
    routes while the backend is released or being released / reacquired.

    Reads the state the main process writes (SharedBackendState), so a request
    is refused without an IPC round trip.
    """

    def __init__(
        self, app: ASGIApp, state: SharedBackendState, retry_after: int
    ) -> None:
        self._app = app
        self._state = state
        self._retry_after = str(retry_after)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] in GATED_PATHS:
            state = self._state.get()
            if state in _UNAVAILABLE:
                await self._unavailable(state)(scope, receive, send)
                return
        await self._app(scope, receive, send)

    def _unavailable(self, state: BackendState) -> Any:
        message = f"Backend {state.label}; retry later"
        return JSONResponse(
            {
                "error": {
                    "message": message,
                    "type": "server_error",
                    "code": "unavailable",
                }
            },
            status_code=503,
            headers={"Retry-After": self._retry_after},
        )
