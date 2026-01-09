"""OpenAI-compatible error formatting."""

from typing import Literal

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """OpenAI error detail object."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI-style error response."""

    error: ErrorDetail


ErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "rate_limit_error",
    "server_error",
    "service_unavailable",
]


def http_status_to_error_type(status_code: int) -> ErrorType:
    """Map HTTP status code to OpenAI error type."""
    mapping: dict[int, ErrorType] = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "server_error",
        503: "service_unavailable",
    }
    return mapping.get(status_code, "server_error")


def create_error_response(
    status_code: int,
    message: str,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    """Create an OpenAI-formatted error response."""
    error_type = http_status_to_error_type(status_code)
    response = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code,
        )
    )
    return JSONResponse(status_code=status_code, content=response.model_dump())


async def openai_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """
    Exception handler that converts HTTPException to OpenAI error format.

    Only applies to /v1/* routes.
    """
    if request.url.path.startswith("/v1/"):
        return create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            code=str(exc.status_code),
        )
    # Default FastAPI behavior for non-OpenAI routes
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


class OpenAIHTTPException(HTTPException):
    """HTTPException with OpenAI-specific fields."""

    def __init__(
        self,
        status_code: int,
        message: str,
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.param = param
        self.code = code
