"""Exception handling for the inference server.

Provides clean logging for known exceptions without full stack traces.
"""

from appinfra.app.fastapi import ExceptionHandler as BaseExceptionHandler
from fastapi import Request
from fastapi.responses import JSONResponse


class ExceptionHandler(BaseExceptionHandler):
    """Handles all exceptions - intercepts known types, re-raises others."""

    async def handle(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle exception - intercept known types, re-raise others."""
        assert self._lg is not None  # Injected after unpickling

        # Trace-level exception details for debugging
        self._lg.trace(
            "exception handler invoked",
            extra={
                "exc_type": type(exc).__name__,
                "exc_module": type(exc).__module__,
                "exc_mro": [c.__name__ for c in type(exc).__mro__],
                "exception": exc,
            },
        )

        if isinstance(exc, TimeoutError):
            return self._handle_timeout(exc)

        # Unknown exception - re-raise for default handling
        raise exc

    def _handle_timeout(self, exc: TimeoutError) -> JSONResponse:
        """Handle request timeout with clean logging."""
        assert self._lg is not None
        self._lg.error("request timeout", extra={"error": str(exc)})
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "code": "timeout",
                }
            },
        )
