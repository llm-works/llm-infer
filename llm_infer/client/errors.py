"""Exception hierarchy for LLM client.

All backend-specific errors and routing errors are translated to these
exceptions, providing a consistent error interface.
"""

from __future__ import annotations


class BackendError(Exception):
    """Base exception for all backend errors.

    All backend implementations translate their native exceptions to this
    hierarchy, enabling consistent error handling across different backends.
    """


class ConfigurationError(Exception):
    """Base exception for configuration errors.

    Raised when client configuration is invalid or inconsistent.
    """


class ModelConflictError(ConfigurationError):
    """Same model found in multiple backends.

    Raised when the model routing table has a conflict - the same model ID
    is configured in multiple backends, making routing ambiguous.

    Attributes:
        model: The conflicting model ID.
        backend1: First backend containing the model.
        backend2: Second backend containing the model.
    """

    def __init__(self, model: str, backend1: str, backend2: str) -> None:
        self.model = model
        self.backend1 = backend1
        self.backend2 = backend2
        super().__init__(
            f"Model '{model}' found in multiple backends: '{backend1}' and '{backend2}'"
        )


class BackendUnavailableError(BackendError):
    """Backend is unreachable.

    Raised when the connection to the backend fails, including:
    - Connection refused (server not running)
    - DNS resolution failure
    - Network unreachable
    """


class BackendTimeoutError(BackendError):
    """Request to the backend timed out.

    Raised when the backend doesn't respond within the configured timeout.
    This can occur during connection, sending the request, or waiting for
    the response.
    """


class BackendRequestError(BackendError):
    """Backend returned an error response.

    Raised when the backend returns an HTTP 4xx or 5xx status code,
    or when the response cannot be parsed correctly.

    Attributes:
        status_code: The HTTP status code, if available.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize the error.

        Args:
            message: Error description.
            status_code: HTTP status code from the backend, if available.
        """
        super().__init__(message)
        self.status_code = status_code
