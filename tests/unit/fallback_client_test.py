"""Unit tests for FallbackClient and fallback helpers."""

from unittest.mock import MagicMock

import pytest

from llm_infer.client.backends import RetryConfig
from llm_infer.client.errors import BackendRequestError, BackendTimeoutError
from llm_infer.client.fallback import FallbackClient
from llm_infer.client.fallback_helper import detect_cycles
from llm_infer.client.router import ResolvedTarget
from llm_infer.client.types import ChatResponse

pytestmark = pytest.mark.unit


class TestDetectCycles:
    """Tests for detect_cycles helper."""

    def test_no_cycles_returns_empty(self) -> None:
        """Config without cycles returns empty set."""
        fallbacks = {
            "gpt-4o": "claude-sonnet",
            "claude-sonnet": "gemini-pro",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == set()
        lg.warning.assert_not_called()

    def test_simple_cycle_detected(self) -> None:
        """Simple A->B->A cycle is detected."""
        fallbacks = {
            "a": "b",
            "b": "a",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == {"a", "b"}
        lg.warning.assert_called_once()
        call_args = lg.warning.call_args
        assert "cycle" in call_args[1]["extra"]

    def test_longer_cycle_detected(self) -> None:
        """Longer A->B->C->A cycle is detected."""
        fallbacks = {
            "a": "b",
            "b": "c",
            "c": "a",
        }
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert cycles == {"a", "b", "c"}
        lg.warning.assert_called_once()

    def test_self_loop_detected(self) -> None:
        """Self-loop A->A is detected."""
        fallbacks = {"a": "a"}
        lg = MagicMock()
        cycles = detect_cycles(fallbacks, lg)
        assert "a" in cycles
        lg.warning.assert_called_once()


class TestFallbackClientImport:
    """Test FallbackClient can be imported."""

    def test_import_from_client_package(self) -> None:
        """FallbackClient is exported from client package."""
        from llm_infer.client import FallbackClient

        assert FallbackClient is not None

    def test_import_directly(self) -> None:
        """FallbackClient can be imported directly."""
        from llm_infer.client.fallback import FallbackClient

        assert FallbackClient is not None


class TestFallbackClientLogging:
    """Tests for FallbackClient logging behavior."""

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Create a mock router with resolve and get_client methods."""
        router = MagicMock()

        # resolve() returns ResolvedTarget with model and backend
        def mock_resolve(model: str | None = None, backend: str | None = None):
            return ResolvedTarget(
                model=model or "default-model", backend="test-backend"
            )

        router.resolve = mock_resolve
        return router

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Create a mock logger."""
        return MagicMock()

    def test_logs_warning_on_fallback(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Should log warning with details when falling back to another model."""
        # Setup: first model fails with 500, second succeeds
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Internal Server Error", status_code=500)
            return ChatResponse(
                content="success", model="claude-sonnet", provider="anthropic"
            )

        mock_client._chat = mock_chat
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act
        response = client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert - response succeeded via fallback
        assert response.content == "success"

        # Assert - warning logged with correct fields
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "model request failed, trying fallback"

        extra = call_args[1]["extra"]
        assert extra["failed_model"] == "gpt-4o"
        assert extra["fallback_model"] == "claude-sonnet"
        assert extra["error_type"] == "BackendRequestError"
        assert extra["status_code"] == 500
        assert "Internal Server Error" in extra["error"]
        assert extra["attempt"] == 1

    def test_logs_multiple_fallbacks_in_chain(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Should log each fallback attempt in a chain."""
        # Setup: first two models fail, third succeeds
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Service Unavailable", status_code=503)
            if call_count == 2:
                raise BackendTimeoutError("Request timed out")
            return ChatResponse(
                content="finally worked", model="gemini-pro", provider="google"
            )

        mock_client._chat = mock_chat
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {
            "gpt-4o": "claude-sonnet",
            "claude-sonnet": "gemini-pro",
        }
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act
        response = client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert - succeeded on third model
        assert response.content == "finally worked"

        # Assert - two warning logs (one per fallback)
        assert mock_logger.warning.call_count == 2

        # First fallback: gpt-4o -> claude-sonnet
        first_call = mock_logger.warning.call_args_list[0]
        assert first_call[1]["extra"]["failed_model"] == "gpt-4o"
        assert first_call[1]["extra"]["fallback_model"] == "claude-sonnet"
        assert first_call[1]["extra"]["status_code"] == 503
        assert first_call[1]["extra"]["attempt"] == 1

        # Second fallback: claude-sonnet -> gemini-pro
        second_call = mock_logger.warning.call_args_list[1]
        assert second_call[1]["extra"]["failed_model"] == "claude-sonnet"
        assert second_call[1]["extra"]["fallback_model"] == "gemini-pro"
        assert second_call[1]["extra"]["error_type"] == "BackendTimeoutError"
        assert second_call[1]["extra"]["status_code"] is None  # Timeout has no status
        assert second_call[1]["extra"]["attempt"] == 2

    def test_logs_error_when_all_models_fail(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Should log error when entire fallback chain is exhausted."""
        # Setup: all models fail
        mock_client = MagicMock()
        mock_client._chat = MagicMock(
            side_effect=BackendRequestError("Server Error", status_code=500)
        )
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act & Assert - should raise after exhausting chain
        with pytest.raises(BackendRequestError):
            client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert - warning for fallback attempt + error for chain exhaustion
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_called_once()

        error_call = mock_logger.error.call_args
        assert error_call[0][0] == "all fallback models failed"
        assert error_call[1]["extra"]["original_model"] == "gpt-4o"
        assert "Server Error" in error_call[1]["extra"]["final_error"]

    def test_no_logging_when_first_model_succeeds(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Should not log anything when first model succeeds."""
        mock_client = MagicMock()
        mock_client._chat = MagicMock(
            return_value=ChatResponse(
                content="success", model="gpt-4o", provider="openai"
            )
        )
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act
        response = client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert
        assert response.content == "success"
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_fallback_on_rate_limit(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """429 triggers fallback: by this layer the inner retry is exhausted."""
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Rate limited", status_code=429)
            return ChatResponse(
                content="success", model="claude-sonnet", provider="anthropic"
            )

        mock_client._chat = mock_chat
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        # Act
        response = client.chat([{"role": "user", "content": "hello"}], model="gpt-4o")

        # Assert - served by the fallback model
        assert response.content == "success"
        assert call_count == 2

        # Assert - fallback logged with the 429
        mock_logger.warning.assert_called_once()
        extra = mock_logger.warning.call_args[1]["extra"]
        assert extra["failed_model"] == "gpt-4o"
        assert extra["fallback_model"] == "claude-sonnet"
        assert extra["status_code"] == 429
        mock_logger.error.assert_not_called()

    def test_cyclic_fallback_retries_until_success(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Cyclic fallback retries round-robin until one model succeeds."""
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(request):
            nonlocal call_count
            call_count += 1
            if call_count < 4:  # Fail first 3 attempts (a, b, a)
                raise BackendRequestError("Service Unavailable", status_code=503)
            return ChatResponse(content="success", model="b", provider="test")

        mock_client._chat = mock_chat
        mock_router.get_client = MagicMock(return_value=mock_client)

        # Cyclic fallback: a -> b -> a (round-robin)
        fallbacks = {"a": "b", "b": "a"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        response = client.chat([{"role": "user", "content": "hello"}], model="a")

        assert response.content == "success"
        assert call_count == 4  # a fails, b fails, a fails, b succeeds


class TestFallbackOnRateLimitAllPaths:
    """429 → fallback across stream/async paths (sync chat covered above)."""

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Create a mock router with resolve and get_client methods."""
        router = MagicMock()

        def mock_resolve(model: str | None = None, backend: str | None = None):
            return ResolvedTarget(
                model=model or "default-model", backend="test-backend"
            )

        router.resolve = mock_resolve
        return router

    @pytest.fixture
    def mock_logger(self) -> MagicMock:
        """Create a mock logger."""
        return MagicMock()

    def test_stream_fallback_on_rate_limit(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Pre-token 429 on a sync stream is served by the fallback model."""
        mock_client = MagicMock()
        call_count = 0

        def mock_chat_stream(request, holder):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Rate limited", status_code=429)
            return iter(["hello", " world"])

        mock_client._chat_stream = mock_chat_stream
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        stream = client.chat_stream(
            [{"role": "user", "content": "hello"}], model="gpt-4o"
        )
        tokens = list(stream)

        assert tokens == ["hello", " world"]
        assert call_count == 2
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[1]["extra"]["status_code"] == 429

    @pytest.mark.asyncio
    async def test_async_fallback_on_rate_limit(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """429 on async chat is served by the fallback model."""
        mock_client = MagicMock()
        call_count = 0

        async def mock_chat_async(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Rate limited", status_code=429)
            return ChatResponse(
                content="success", model="claude-sonnet", provider="anthropic"
            )

        mock_client._chat_async = mock_chat_async
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        response = await client.chat_async(
            [{"role": "user", "content": "hello"}], model="gpt-4o"
        )

        assert response.content == "success"
        assert call_count == 2
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[1]["extra"]["status_code"] == 429

    @pytest.mark.asyncio
    async def test_async_stream_fallback_on_rate_limit(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Pre-token 429 on an async stream is served by the fallback model."""
        mock_client = MagicMock()
        call_count = 0

        def mock_chat_stream_async(request, holder):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendRequestError("Rate limited", status_code=429)

            async def gen():
                yield "hello"
                yield " world"

            return gen()

        mock_client._chat_stream_async = mock_chat_stream_async
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        stream = client.chat_stream_async(
            [{"role": "user", "content": "hello"}], model="gpt-4o"
        )
        tokens = [token async for token in stream]

        assert tokens == ["hello", " world"]
        assert call_count == 2
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[1]["extra"]["status_code"] == 429

    def test_mid_stream_rate_limit_propagates(
        self, mock_router: MagicMock, mock_logger: MagicMock
    ) -> None:
        """429 after the first token propagates: partial output can't be replayed."""

        def mock_chat_stream(request, holder):
            yield "partial"
            raise BackendRequestError("Rate limited", status_code=429)

        mock_client = MagicMock()
        mock_client._chat_stream = mock_chat_stream
        mock_router.get_client = MagicMock(return_value=mock_client)

        fallbacks = {"gpt-4o": "claude-sonnet"}
        client = FallbackClient(mock_logger, mock_router, fallbacks)

        stream = client.chat_stream(
            [{"role": "user", "content": "hello"}], model="gpt-4o"
        )
        tokens = []
        with pytest.raises(BackendRequestError) as exc_info:
            for token in stream:
                tokens.append(token)

        assert tokens == ["partial"]
        assert exc_info.value.status_code == 429
        mock_logger.warning.assert_not_called()


class TestNoRetryWarning:
    """Construction-time warning for backends without retry config."""

    def _router(self, retry: RetryConfig | None) -> MagicMock:
        """Mock router exposing a real clients mapping with the given retry."""
        router = MagicMock()
        client = MagicMock()
        client.backend.ctx.retry = retry
        router.clients = {"primary": client}
        return router

    def test_warns_when_backend_has_no_retry(self) -> None:
        """retry: None means fallback engages on the first transient error."""
        lg = MagicMock()
        FallbackClient(lg, self._router(None), {"a": "b"})

        lg.warning.assert_called_once()
        msg = lg.warning.call_args[0][0]
        assert "no retry config" in msg
        assert lg.warning.call_args[1]["extra"]["backend"] == "primary"

    def test_no_warning_when_retry_configured(self) -> None:
        """Backends with a retry budget construct silently."""
        lg = MagicMock()
        FallbackClient(lg, self._router(RetryConfig()), {"a": "b"})

        lg.warning.assert_not_called()
