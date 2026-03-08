"""Centralized error response mapping.

Provides consistent HTTP error responses for internal request statuses.
"""

from typing import Any, TypeVar

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from ..dispatch.types import RequestStatus

T = TypeVar("T")

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


def get_http_status_for_request_status(status: RequestStatus) -> tuple[int, str] | None:
    """Get HTTP status code and default message for a RequestStatus.

    Args:
        status: The internal request status.

    Returns:
        Tuple of (status_code, default_message) or None if not an error status.
    """
    return _ERROR_MAPPINGS.get(status)


async def submit_or_timeout(
    ipc: Any, request_id: str, request: Any
) -> Any | JSONResponse:
    """Submit request via IPC, returning 504 JSONResponse on timeout.

    Args:
        ipc: The IPC client instance.
        request_id: Unique request identifier.
        request: The request object to submit.

    Returns:
        The response from the IPC call, or a 504 JSONResponse on timeout.
    """
    try:
        return await ipc.submit(request_id, request)
    except TimeoutError as e:
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
