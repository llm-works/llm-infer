# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Tell the engine side to stop work on a request the API side gave up on."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from appinfra.log import Logger
from starlette.requests import Request as HTTPRequest

from ..dispatch.types import CancelRequest

# How long the cancel waits for the engine side's acknowledgement
CANCEL_ACK_TIMEOUT = 5.0

# The cancel goes through the same IPC channel as requests, which refuses new
# messages while its pending limit is reached. Slots free up quickly (the
# handler answers requests beyond its own, smaller limit with 503 at once), so
# the cancel retries until RETRY_WINDOW has passed.
CANCEL_RETRY_INTERVAL = 0.1
CANCEL_RETRY_WINDOW = 30.0

# Strong references: the event loop keeps only weak ones to running tasks
_pending: set[asyncio.Task[None]] = set()


def send_cancel(lg: Logger, ipc: Any, request_id: str) -> None:
    """Cancel request_id on the engine side without waiting.

    Safe to call while the calling task is being cancelled (client
    disconnect): the cancel runs in its own task.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # e.g. a stream generator finalized outside the loop
        lg.debug("no event loop to send cancel", extra={"request_id": request_id})
        return
    task = loop.create_task(_cancel(lg, ipc, request_id))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def _cancel(lg: Logger, ipc: Any, request_id: str) -> None:
    message = CancelRequest(id=f"cancel-{uuid.uuid4().hex}", request_id=request_id)
    deadline = asyncio.get_running_loop().time() + CANCEL_RETRY_WINDOW
    while True:
        try:
            ack = await ipc.submit(message, timeout=CANCEL_ACK_TIMEOUT)
            break
        except RuntimeError as e:  # IPCChannel's pending limit; channel errors aren't
            if asyncio.get_running_loop().time() >= deadline:
                _log_undelivered(lg, request_id, e)
                return
            await asyncio.sleep(CANCEL_RETRY_INTERVAL)
        except Exception as e:
            _log_undelivered(lg, request_id, e)
            return
    lg.debug("request cancelled", extra={"request_id": request_id, "found": ack.found})


def _log_undelivered(lg: Logger, request_id: str, e: Exception) -> None:
    lg.warning("cancel not delivered", extra={"request_id": request_id, "exception": e})


async def wait_for_disconnect(http_request: HTTPRequest) -> None:
    """Return once the client has disconnected.

    Consumes the request's remaining ASGI messages, so the body must have been
    read already (FastAPI has parsed it into the endpoint's model).
    """
    while True:
        message = await http_request.receive()
        if message["type"] == "http.disconnect":
            return
