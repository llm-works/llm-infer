"""Request/response trace middleware for debugging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

if TYPE_CHECKING:
    from appinfra.log import Logger


class TraceMiddleware(BaseHTTPMiddleware):
    """Middleware that logs request/response at TRACE level."""

    async def _log_request(self, lg: Logger, request: Request) -> None:
        """Log request body."""
        try:
            body = await request.body()
            if body:
                lg.trace(
                    "request", extra={"path": request.url.path, "body": body.decode()}
                )
        except Exception as e:
            lg.warning("trace middleware error", extra={"exception": e})

    async def _log_response(
        self, lg: Logger, request: Request, response: StreamingResponse
    ) -> Response:
        """Log response body and return reconstructed response."""
        body = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                body += chunk
            elif isinstance(chunk, str):
                body += chunk.encode()
            else:
                body += bytes(chunk)
        try:
            lg.trace(
                "response",
                extra={
                    "path": request.url.path,
                    "status": response.status_code,
                    "body": body.decode(),
                },
            )
        except Exception:
            pass
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request and log at trace level."""
        lg: Logger | None = getattr(request.app.state, "lg", None)
        is_api = lg is not None and request.url.path.startswith("/v1/")

        if is_api:
            await self._log_request(cast("Logger", lg), request)

        response = await call_next(request)

        if is_api:
            return await self._log_response(cast("Logger", lg), request, response)

        return cast(Response, response)
