# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""The admin listener: a listener of its own in serve's main process, never on the
public API port, bound to an intranet address, with its own bearer tokens.

* `POST /admin/release`: stop the backend, freeing its GPU memory. Returns 200
  once its processes are gone (`{"state": "released"}`); inference answers 503
  meanwhile and until reacquired.
* `POST /admin/reacquire`: rescan the LoRA adapter directory and start the
  backend again with what it finds. 202 with `{"state": "loading"}`; 200 if
  already serving. Follow progress with status.
* `GET /admin/status`: `{"state": ..., "last_error": ...}`, state one of
  initializing, serving, releasing, released, loading.

Every call needs `Authorization: Bearer <token>` with a configured token (401
otherwise). A call the backend's state doesn't allow, or an engine that can't
release, is answered 409.
"""

from __future__ import annotations

import asyncio
import hmac
import threading

from appinfra.app.fastapi import Server, ServerBuilder
from appinfra.log import Logger
from fastapi import Request
from fastapi.responses import JSONResponse

from ..dispatch.config import AdminConfig
from .controller import AdminError, BackendController
from .state import BackendState


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


class AdminApi:
    """The handlers; see the module docstring."""

    def __init__(self, lg: Logger, controller: BackendController, cfg: AdminConfig):
        self._lg = lg
        self._controller = controller
        self._tokens = [t.encode() for t in cfg.tokens]

    def _authorized(self, request: Request) -> bool:
        """Whether the Authorization header carries a configured bearer token.

        Every token is compared, in constant time, so timing doesn't tell which
        one nearly matched.
        """
        scheme, _, presented = (request.headers.get("authorization") or "").partition(
            " "
        )
        if scheme.lower() != "bearer" or not presented.strip():
            return False
        candidate = presented.strip().encode()
        found = False
        for token in self._tokens:
            found |= hmac.compare_digest(token, candidate)
        return found

    async def status(self, request: Request) -> JSONResponse:
        if not self._authorized(request):
            return _error(401, "unauthorized")
        return JSONResponse(self._controller.status())

    async def release(self, request: Request) -> JSONResponse:
        if not self._authorized(request):
            return _error(401, "unauthorized")
        try:
            # Blocks for the backend's stop (seconds): off the event loop
            state = await asyncio.to_thread(self._controller.release)
        except AdminError as e:
            return _error(e.status_code, str(e))
        return JSONResponse({"state": state.label})

    async def reacquire(self, request: Request) -> JSONResponse:
        if not self._authorized(request):
            return _error(401, "unauthorized")
        try:
            state = self._controller.start_reacquire()
        except AdminError as e:
            return _error(e.status_code, str(e))
        code = 200 if state == BackendState.SERVING else 202
        return JSONResponse({"state": state.label}, status_code=code)


STARTUP_SETTLE_TIME = 0.5


class AdminStartupError(Exception):
    """The admin listener failed to start (e.g., port already in use)."""


def _build_server(lg: Logger, api: AdminApi, cfg: AdminConfig) -> Server:
    return (
        ServerBuilder(lg, "admin")
        .with_host(cfg.host)
        .with_port(cfg.port)
        .routes.with_route("/admin/status", api.status, methods=["GET"])
        .with_route("/admin/release", api.release, methods=["POST"])
        .with_route("/admin/reacquire", api.reacquire, methods=["POST"])
        .done()
        .build()
    )


def start(
    lg: Logger, controller: BackendController, cfg: AdminConfig
) -> threading.Thread:
    """Serve the admin API from a daemon thread; it ends with the process.

    Uvicorn installs no signal handlers outside the main thread, so serve's own
    SIGINT/SIGTERM handling stays in place.

    Raises:
        AdminStartupError: If uvicorn fails to bind (e.g., port already in use).
    """
    api = AdminApi(lg, controller, cfg)
    server = _build_server(lg, api, cfg)
    startup_error: list[BaseException] = []
    failed = threading.Event()

    def run() -> None:
        try:
            server.start()
        except BaseException as e:  # uvicorn exits with SystemExit when it can't bind
            startup_error.append(e)
            failed.set()
            lg.error(
                "admin listener failed",
                extra={"host": cfg.host, "port": cfg.port, "exception": e},
            )

    thread = threading.Thread(target=run, name="admin-api", daemon=True)
    thread.start()
    if failed.wait(STARTUP_SETTLE_TIME):
        raise AdminStartupError(
            f"admin listener failed to bind {cfg.host}:{cfg.port}"
        ) from startup_error[0]
    lg.info(
        "admin listener started",
        extra={"host": cfg.host, "port": cfg.port, "tokens": len(cfg.tokens)},
    )
    if not cfg.tokens:
        lg.warning("admin.tokens is empty; every admin call is refused with 401")
    return thread
