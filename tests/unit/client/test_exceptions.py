"""Unit tests for client exceptions."""

import pytest

from llm_infer.client import (
    BackendError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)

pytestmark = pytest.mark.unit


class TestBackendError:
    """Test BackendError base exception."""

    def test_is_exception(self) -> None:
        """Test BackendError is an Exception."""
        assert issubclass(BackendError, Exception)

    def test_can_raise_with_message(self) -> None:
        """Test can raise with message."""
        with pytest.raises(BackendError, match="test error"):
            raise BackendError("test error")


class TestBackendUnavailableError:
    """Test BackendUnavailableError exception."""

    def test_inherits_from_backend_error(self) -> None:
        """Test inherits from BackendError."""
        assert issubclass(BackendUnavailableError, BackendError)

    def test_can_raise_with_message(self) -> None:
        """Test can raise with message."""
        with pytest.raises(BackendUnavailableError, match="connection refused"):
            raise BackendUnavailableError("connection refused")


class TestBackendTimeoutError:
    """Test BackendTimeoutError exception."""

    def test_inherits_from_backend_error(self) -> None:
        """Test inherits from BackendError."""
        assert issubclass(BackendTimeoutError, BackendError)

    def test_can_raise_with_message(self) -> None:
        """Test can raise with message."""
        with pytest.raises(BackendTimeoutError, match="timed out"):
            raise BackendTimeoutError("request timed out")


class TestBackendRequestError:
    """Test BackendRequestError exception."""

    def test_inherits_from_backend_error(self) -> None:
        """Test inherits from BackendError."""
        assert issubclass(BackendRequestError, BackendError)

    def test_has_status_code_attribute(self) -> None:
        """Test status_code attribute is stored."""
        error = BackendRequestError("bad request", status_code=400)
        assert error.status_code == 400

    def test_status_code_defaults_to_none(self) -> None:
        """Test status_code defaults to None."""
        error = BackendRequestError("error")
        assert error.status_code is None

    def test_message_is_accessible(self) -> None:
        """Test message is accessible via str()."""
        error = BackendRequestError("bad request", status_code=400)
        assert str(error) == "bad request"
