# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Centralized error response mapping.

Provides consistent HTTP error responses for internal request statuses.
"""

from typing import Any

from appinfra.log import Logger
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from ..dispatch.types import RequestStatus

# Status -> (HTTP code, default message)
_ERROR_MAPPINGS: dict[RequestStatus, tuple[int, str]] = {
    RequestStatus.REJECTED: (503, "Server at capacity"),
    RequestStatus.FAILED: (500, "Internal error"),
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


async def submit_or_timeout(lg: Logger, ipc: Any, request: Any) -> Any | JSONResponse:
    """Submit request via IPC, returning 504 JSONResponse on timeout.

    Args:
        lg: Logger for recording timeout errors.
        ipc: The IPC client instance.
        request: The request object to submit (must have .id attribute).

    Returns:
        The response from the IPC call, or a 504 JSONResponse on timeout.
    """
    try:
        return await ipc.submit(request)
    except TimeoutError as e:
        lg.warning("IPC timeout", extra={"request_id": request.id, "error": str(e)})
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "timeout",
                }
            },
        )
