"""Centralized error response mapping.

Provides consistent HTTP error responses for internal request statuses.
"""

from typing import Any

from fastapi import HTTPException

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


def get_http_status_for_request_status(status: RequestStatus) -> tuple[int, str] | None:
    """Get HTTP status code and default message for a RequestStatus.

    Args:
        status: The internal request status.

    Returns:
        Tuple of (status_code, default_message) or None if not an error status.
    """
    return _ERROR_MAPPINGS.get(status)
