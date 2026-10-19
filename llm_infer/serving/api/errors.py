# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Centralized error response mapping.

Provides consistent HTTP error responses for internal request statuses.
"""

import asyncio
from typing import Any

from appinfra.log import Logger
from appinfra.service.errors import ChannelClosedError, ChannelError
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.requests import Request as HTTPRequest

from ..dispatch.types import RequestStatus
from .cancel import send_cancel, wait_for_disconnect

# Status -> (HTTP code, default message)
#
# appinfra's channel raises ChannelError for any response whose `error` field is
# set, so a response only reaches raise_for_error_status() with `error` unset.
# Statuses that must keep their own HTTP code are sent without `error` and rely
# on the default message here.
_ERROR_MAPPINGS: dict[RequestStatus, tuple[int, str]] = {
    RequestStatus.REJECTED: (503, "Server at capacity"),
    RequestStatus.FAILED: (500, "Internal error"),
    RequestStatus.UNAVAILABLE: (503, "Server shutting down"),
}


def raise_for_error_status(response: Any) -> None:
    """Raise HTTPException if response indicates an error.

    Args:
        response: A response object with `status` and optional `error` attributes.

    Raises:
        HTTPException: If the response status indicates an error.
    """
    if response.status not in _ERROR_MAPPINGS:
        return

    status_code, default_message = _ERROR_MAPPINGS[response.status]
    detail = response.error or default_message
    raise HTTPException(status_code=status_code, detail=detail)


def log_for_error_status(lg: Logger, response: Any) -> None:
    """Log a warning when a response carries an error status.

    Companion to raise_for_error_status for response types whose success
    payload does not carry a status field (e.g. MetricsResponse). A future
    variant that surfaces a failed status here is logged for observability
    rather than raised, keeping the success path intact.
    """
    status = getattr(response, "status", None)
    if status not in _ERROR_MAPPINGS:
        return
    _, default_message = _ERROR_MAPPINGS[status]
    error = getattr(response, "error", None) or default_message
    lg.warning(
        "response reports error status",
        extra={"status": str(status), "error": error},
    )


def get_http_status_for_request_status(status: RequestStatus) -> tuple[int, str] | None:
    """Get HTTP status code and default message for a RequestStatus.

    Args:
        status: The internal request status.

    Returns:
        Tuple of (status_code, default_message) or None if not an error status.
    """
    return _ERROR_MAPPINGS.get(status)


def _error_json(status_code: int, message: str, code: str) -> JSONResponse:
    """Build an OpenAI-style error JSONResponse."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "server_error", "code": code}},
    )


async def submit_or_timeout(lg: Logger, ipc: Any, request: Any) -> Any | JSONResponse:
    """Submit request via IPC, mapping channel failures to error responses.

    - timeout: 504
    - channel closed (the server is shutting down): 503
    - any other ChannelError (raised when the response carries `error`): 500
      with the error text

    Args:
        lg: Logger for recording IPC failures.
        ipc: The IPC client instance.
        request: The request object to submit (must have .id attribute).

    Returns:
        The response from the IPC call, or an error JSONResponse.
    """
    try:
        return await ipc.submit(request)
    except (TimeoutError, ChannelError) as e:
        return _ipc_error_response(lg, request.id, e)


def _ipc_error_response(lg: Logger, request_id: str, e: Exception) -> JSONResponse:
    """Map a failed IPC submit to an error response (see submit_or_timeout)."""
    if isinstance(e, TimeoutError):  # ChannelTimeoutError is a TimeoutError
        lg.warning("IPC timeout", extra={"request_id": request_id, "error": str(e)})
        return _error_json(504, str(e), "timeout")
    if isinstance(e, ChannelClosedError):
        lg.warning("IPC closed", extra={"request_id": request_id, "error": str(e)})
        return _error_json(503, "Server shutting down", "unavailable")
    lg.warning("request failed", extra={"request_id": request_id, "error": str(e)})
    return _error_json(500, str(e), "error")


async def submit_inference(
    lg: Logger, ipc: Any, request: Any, http_request: HTTPRequest
) -> Any | JSONResponse:
    """submit_or_timeout for an inference request, cancelled when the caller gives up.

    On timeout (504) or client disconnect, the engine side is told to stop
    work on the request, so it no longer holds a handler slot and the backend.
    """
    submit = asyncio.ensure_future(ipc.submit(request))
    disconnect = asyncio.ensure_future(wait_for_disconnect(http_request))
    try:
        await asyncio.wait({submit, disconnect}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        submit.cancel()
        send_cancel(lg, ipc, request.id)
        raise
    finally:
        disconnect.cancel()

    if not submit.done():
        submit.cancel()
        lg.info("client disconnected", extra={"request_id": request.id})
        send_cancel(lg, ipc, request.id)
        return _error_json(499, "Client disconnected", "client_closed")
    try:
        return submit.result()
    except (TimeoutError, ChannelError) as e:
        if isinstance(e, TimeoutError):
            send_cancel(lg, ipc, request.id)
        return _ipc_error_response(lg, request.id, e)
