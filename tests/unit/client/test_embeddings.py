"""Unit tests for EmbeddingClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import (
    BackendRequestError,
    BackendUnavailableError,
    EmbeddingCallbacks,
    EmbeddingClient,
    EmbeddingRequest,
    EmbeddingResult,
    RetryConfig,
)
from llm_infer.client.backends.embedding import (
    Backend,
    BatchEmbeddingResult,
    OpenAIBackend,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


@pytest.fixture
def mock_backend(mock_lg: Logger) -> MagicMock:
    """Create a mock embedding backend."""
    backend = MagicMock(spec=Backend)
    backend.model = "test-model"
    backend.provider = "test-provider"
    return backend


class TestEmbeddingClientInit:
    """Test EmbeddingClient initialization."""

    def test_basic_init(self, mock_lg: Logger, mock_backend: MagicMock) -> None:
        """Test basic initialization."""
        client = EmbeddingClient(mock_lg, mock_backend)
        assert client.model == "test-model"
        assert client.backend is mock_backend
        client.close()

    def test_with_retry_config(self, mock_lg: Logger, mock_backend: MagicMock) -> None:
        """Test initialization with retry config."""
        retry = RetryConfig(base=1.0, factor=2.0, timeout=60.0)
        client = EmbeddingClient(mock_lg, mock_backend, retry=retry)
        assert client._retry is retry
        client.close()


class TestEmbeddingClientEmbed:
    """Test embed method."""

    def test_embed_delegates_to_backend(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test embed delegates to backend."""
        expected = EmbeddingResult(
            embedding=[0.1, 0.2], model="model", dimensions=2, prompt_tokens=5
        )
        mock_backend.embed.return_value = expected

        client = EmbeddingClient(mock_lg, mock_backend)
        result = client.embed("hello")

        assert result is expected
        mock_backend.embed.assert_called_once_with("hello", model=None, dimensions=None)
        client.close()

    def test_embed_batch_delegates_to_backend(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test embed_batch delegates to backend."""

        expected = BatchEmbeddingResult(
            embeddings=[[0.1], [0.2]],
            model="model",
            dimensions=1,
            size=2,
            total_prompt_tokens=10,
        )
        mock_backend.embed_batch.return_value = expected

        client = EmbeddingClient(mock_lg, mock_backend)
        result = client.embed_batch(["a", "b"])

        assert result is expected
        assert result.total_prompt_tokens == 10
        mock_backend.embed_batch.assert_called_once_with(
            ["a", "b"], model=None, dimensions=None
        )
        client.close()

    def test_embed_batch_empty_list(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test embed_batch with empty list doesn't call backend."""
        client = EmbeddingClient(mock_lg, mock_backend)
        result = client.embed_batch([])

        assert result.embeddings == []
        mock_backend.embed_batch.assert_not_called()
        client.close()

    def test_client_level_defaults_forwarded(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test client-level model and dimensions defaults are forwarded to backend."""
        expected = EmbeddingResult(
            embedding=[0.1, 0.2],
            model="override-model",
            dimensions=256,
            prompt_tokens=5,
        )
        mock_backend.embed.return_value = expected

        client = EmbeddingClient(
            mock_lg, mock_backend, model="override-model", dimensions=256
        )
        result = client.embed("hello")

        assert result is expected
        mock_backend.embed.assert_called_once_with(
            "hello", model="override-model", dimensions=256
        )
        client.close()

    def test_per_call_overrides_client_defaults(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test per-call parameters override client-level defaults."""
        expected = EmbeddingResult(
            embedding=[0.1], model="call-model", dimensions=128, prompt_tokens=3
        )
        mock_backend.embed.return_value = expected

        client = EmbeddingClient(
            mock_lg, mock_backend, model="client-model", dimensions=256
        )
        result = client.embed("hello", model="call-model", dimensions=128)

        assert result is expected
        mock_backend.embed.assert_called_once_with(
            "hello", model="call-model", dimensions=128
        )
        client.close()


class TestEmbeddingClientRetry:
    """Test retry behavior."""

    def test_retry_on_transient_error(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test retry on transient errors."""
        retry = RetryConfig(base=0.01, factor=1.0, timeout=10.0)
        client = EmbeddingClient(mock_lg, mock_backend, retry=retry)

        expected = EmbeddingResult(
            embedding=[0.1], model="model", dimensions=1, prompt_tokens=5
        )
        call_count = 0

        def side_effect(text, *, model=None, dimensions=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise BackendRequestError("Service unavailable", status_code=503)
            return expected

        mock_backend.embed.side_effect = side_effect

        result = client.embed("test")

        assert call_count == 3
        assert result is expected
        client.close()

    def test_no_retry_on_client_error(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test no retry on 4xx client errors."""
        retry = RetryConfig(base=0.01, factor=1.0, timeout=10.0)
        client = EmbeddingClient(mock_lg, mock_backend, retry=retry)

        mock_backend.embed.side_effect = BackendRequestError(
            "Bad request", status_code=400
        )

        with pytest.raises(BackendRequestError) as exc_info:
            client.embed("test")

        assert exc_info.value.status_code == 400
        mock_backend.embed.assert_called_once()
        client.close()

    def test_no_retry_when_disabled(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test no retry when retry config is None."""
        client = EmbeddingClient(mock_lg, mock_backend, retry=None)

        mock_backend.embed.side_effect = BackendRequestError(
            "Service unavailable", status_code=503
        )

        with pytest.raises(BackendRequestError):
            client.embed("test")

        mock_backend.embed.assert_called_once()
        client.close()

    def test_retry_on_unavailable_error(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test retry on BackendUnavailableError."""
        retry = RetryConfig(base=0.01, factor=1.0, timeout=10.0)
        client = EmbeddingClient(mock_lg, mock_backend, retry=retry)

        expected = EmbeddingResult(
            embedding=[0.1], model="model", dimensions=1, prompt_tokens=5
        )
        call_count = 0

        def side_effect(text, *, model=None, dimensions=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise BackendUnavailableError("Connection refused")
            return expected

        mock_backend.embed.side_effect = side_effect

        result = client.embed("test")

        assert call_count == 2
        assert result is expected
        client.close()


class TestEmbeddingClientAsync:
    """Test async methods."""

    @pytest.mark.asyncio
    async def test_embed_async_delegates_to_backend(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test embed_async delegates to backend."""
        expected = EmbeddingResult(
            embedding=[0.1], model="model", dimensions=1, prompt_tokens=5
        )

        async def mock_embed_async(text, *, model=None, dimensions=None):
            return expected

        mock_backend.embed_async = mock_embed_async

        client = EmbeddingClient(mock_lg, mock_backend)
        result = await client.embed_async("test")

        assert result is expected
        await client.aclose()

    @pytest.mark.asyncio
    async def test_embed_batch_async_empty(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test embed_batch_async with empty list."""
        client = EmbeddingClient(mock_lg, mock_backend)
        result = await client.embed_batch_async([])

        assert result.embeddings == []
        mock_backend.embed_batch_async.assert_not_called()
        await client.aclose()


class TestEmbeddingClientContextManager:
    """Test context manager support."""

    def test_sync_context_manager(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test sync context manager."""
        with EmbeddingClient(mock_lg, mock_backend) as client:
            assert isinstance(client, EmbeddingClient)
        mock_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test async context manager."""
        mock_backend.aclose = AsyncMock()

        async with EmbeddingClient(mock_lg, mock_backend) as client:
            assert isinstance(client, EmbeddingClient)

        mock_backend.aclose.assert_awaited_once()


class TestEmbeddingClientLogging:
    """Test entry/exit/failure DEBUG logs around embedding calls."""

    @staticmethod
    def _debug_calls(mock_lg: MagicMock) -> list[tuple[str, dict]]:
        """Extract (message, extra) tuples from mock logger's debug calls."""
        out: list[tuple[str, dict]] = []
        for call in mock_lg.debug.call_args_list:
            msg = call.args[0] if call.args else call.kwargs.get("msg", "")
            extra = call.kwargs.get("extra", {})
            out.append((msg, extra))
        return out

    def test_embed_logs_request_and_response(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """embed() emits paired request/response debug logs on success."""
        expected = EmbeddingResult(
            embedding=[0.1], model="m", dimensions=1, prompt_tokens=7
        )
        mock_backend.embed.return_value = expected

        client = EmbeddingClient(mock_lg, mock_backend, model="override")
        client.embed("hello")
        client.close()

        calls = self._debug_calls(mock_lg)
        assert len(calls) == 2
        req_msg, req_extra = calls[0]
        done_msg, done_extra = calls[1]
        assert req_msg == "embedding request..."
        assert done_msg == "embedding response"
        assert req_extra["req"] == done_extra["req"]
        assert req_extra["model"] == "override"
        assert req_extra["backend"] == "test-provider"
        assert done_extra["backend"] == "test-provider"
        assert req_extra["chars"] == 5
        assert done_extra["chars"] == 5
        assert done_extra["tokens"] == 7
        assert isinstance(done_extra["after"], float | int)
        assert done_extra["after"] >= 0

    def test_embed_batch_logs_use_count(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """embed_batch() logs use 'count' and 'total_prompt_tokens'."""

        expected = BatchEmbeddingResult(
            embeddings=[[0.1], [0.2], [0.3]],
            model="m",
            dimensions=1,
            size=3,
            total_prompt_tokens=42,
        )
        mock_backend.embed_batch.return_value = expected

        client = EmbeddingClient(mock_lg, mock_backend)
        client.embed_batch(["a", "b", "c"])
        client.close()

        calls = self._debug_calls(mock_lg)
        assert len(calls) == 2
        assert calls[0][1]["count"] == 3
        assert "chars" not in calls[0][1]
        assert calls[1][1]["count"] == 3
        assert calls[1][1]["tokens"] == 42
        assert calls[0][1]["req"] == calls[1][1]["req"]

    def test_embed_batch_empty_does_not_log(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """Empty batch short-circuit skips logging (no backend call to triage)."""
        client = EmbeddingClient(mock_lg, mock_backend)
        client.embed_batch([])
        client.close()

        assert self._debug_calls(mock_lg) == []

    def test_embed_logs_failed_on_terminal_error(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """embed() emits request/failed pair when the call raises."""
        mock_backend.embed.side_effect = BackendRequestError("boom", status_code=400)

        client = EmbeddingClient(mock_lg, mock_backend)
        with pytest.raises(BackendRequestError):
            client.embed("hello")
        client.close()

        calls = self._debug_calls(mock_lg)
        assert len(calls) == 2
        req_msg, req_extra = calls[0]
        fail_msg, fail_extra = calls[1]
        assert req_msg == "embedding request..."
        assert fail_msg == "embedding failed"
        assert req_extra["req"] == fail_extra["req"]
        assert fail_extra["backend"] == "test-provider"
        assert fail_extra["chars"] == 5
        assert isinstance(fail_extra["exception"], BackendRequestError)
        assert fail_extra["after"] >= 0

    def test_embed_batch_logs_failed_on_terminal_error(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """embed_batch() emits request/failed pair when the call raises."""
        mock_backend.embed_batch.side_effect = BackendRequestError(
            "boom", status_code=400
        )

        client = EmbeddingClient(mock_lg, mock_backend)
        with pytest.raises(BackendRequestError):
            client.embed_batch(["a", "b"])
        client.close()

        calls = self._debug_calls(mock_lg)
        assert len(calls) == 2
        assert calls[0][0] == "embedding request..."
        assert calls[1][0] == "embedding failed"
        assert calls[0][1]["req"] == calls[1][1]["req"]
        assert calls[1][1]["count"] == 2
        assert isinstance(calls[1][1]["exception"], BackendRequestError)

    @pytest.mark.asyncio
    async def test_embed_async_logs_failed_on_terminal_error(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """embed_async() emits request/failed pair when the call raises."""

        async def _embed_async_fail(text, *, model=None, dimensions=None):
            raise BackendRequestError("boom", status_code=400)

        mock_backend.embed_async = _embed_async_fail

        client = EmbeddingClient(mock_lg, mock_backend)
        with pytest.raises(BackendRequestError):
            await client.embed_async("hello")
        await client.aclose()

        calls = self._debug_calls(mock_lg)
        assert len(calls) == 2
        assert calls[0][0] == "embedding request..."
        assert calls[1][0] == "embedding failed"
        assert calls[0][1]["req"] == calls[1][1]["req"]
        assert calls[1][1]["chars"] == 5

    @pytest.mark.asyncio
    async def test_embed_batch_async_logs_failed_on_terminal_error(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """embed_batch_async() emits request/failed pair when the call raises."""

        async def _embed_batch_async_fail(texts, *, model=None, dimensions=None):
            raise BackendRequestError("boom", status_code=400)

        mock_backend.embed_batch_async = _embed_batch_async_fail

        client = EmbeddingClient(mock_lg, mock_backend)
        with pytest.raises(BackendRequestError):
            await client.embed_batch_async(["a", "b", "c"])
        await client.aclose()

        calls = self._debug_calls(mock_lg)
        assert len(calls) == 2
        assert calls[0][0] == "embedding request..."
        assert calls[1][0] == "embedding failed"
        assert calls[0][1]["req"] == calls[1][1]["req"]
        assert calls[1][1]["count"] == 3

    @pytest.mark.asyncio
    async def test_embed_async_logs_request_and_response(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """embed_async() emits paired request/response debug logs on success."""
        expected = EmbeddingResult(
            embedding=[0.1], model="m", dimensions=1, prompt_tokens=9
        )

        async def _embed_async(text, *, model=None, dimensions=None):
            return expected

        mock_backend.embed_async = _embed_async

        client = EmbeddingClient(mock_lg, mock_backend)
        await client.embed_async("hey")
        await client.aclose()

        calls = self._debug_calls(mock_lg)
        assert [c[0] for c in calls] == [
            "embedding request...",
            "embedding response",
        ]
        assert calls[0][1]["chars"] == 3
        assert calls[1][1]["tokens"] == 9
        assert calls[0][1]["req"] == calls[1][1]["req"]

    @pytest.mark.asyncio
    async def test_embed_batch_async_logs_use_count(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """embed_batch_async() logs use 'count' and 'total_prompt_tokens'."""

        expected = BatchEmbeddingResult(
            embeddings=[[0.1], [0.2]],
            model="m",
            dimensions=1,
            size=2,
            total_prompt_tokens=11,
        )

        async def _embed_batch_async(texts, *, model=None, dimensions=None):
            return expected

        mock_backend.embed_batch_async = _embed_batch_async

        client = EmbeddingClient(mock_lg, mock_backend)
        await client.embed_batch_async(["a", "b"])
        await client.aclose()

        calls = self._debug_calls(mock_lg)
        assert len(calls) == 2
        assert calls[0][1]["count"] == 2
        assert calls[1][1]["tokens"] == 11

    def test_req_ids_are_unique_across_calls(
        self, mock_lg: MagicMock, mock_backend: MagicMock
    ) -> None:
        """Each embed call mints a fresh req id so concurrent lines can be paired."""
        expected = EmbeddingResult(
            embedding=[0.1], model="m", dimensions=1, prompt_tokens=1
        )
        mock_backend.embed.return_value = expected

        client = EmbeddingClient(mock_lg, mock_backend)
        client.embed("a")
        client.embed("b")
        client.close()

        calls = self._debug_calls(mock_lg)
        assert calls[0][1]["req"] != calls[2][1]["req"]


class TestEmbeddingClientCallbacks:
    """User context threads onto EmbeddingRequest and callbacks fire around calls."""

    @staticmethod
    def _ok_result() -> EmbeddingResult:
        return EmbeddingResult(
            embedding=[0.1], model="m", dimensions=1, prompt_tokens=7
        )

    @staticmethod
    def _ok_batch(size: int = 2, tokens: int = 11) -> BatchEmbeddingResult:
        return BatchEmbeddingResult(
            embeddings=[[0.1]] * size,
            model="m",
            dimensions=1,
            size=size,
            total_prompt_tokens=tokens,
        )

    def test_context_threads_onto_request(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """context= kwarg lands on EmbeddingRequest.context in on_response."""
        mock_backend.embed.return_value = self._ok_result()
        captured: list[EmbeddingRequest] = []

        def on_response(req: EmbeddingRequest, _resp: EmbeddingResult) -> None:
            captured.append(req)

        client = EmbeddingClient(
            mock_lg, mock_backend, callbacks=EmbeddingCallbacks(on_response=on_response)
        )
        ctx = {"cost_tracker": object(), "operation": "kelt.embed_entity"}
        client.embed("hello", context=ctx)
        client.close()

        assert len(captured) == 1
        assert captured[0].context == ctx
        assert captured[0].texts == ["hello"]
        assert captured[0].model is None  # no override, backend default

    def test_on_response_receives_result_and_request(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """on_response gets (request, result) — the pair a cost tracker needs."""
        expected = self._ok_result()
        mock_backend.embed.return_value = expected
        pairs: list[tuple[EmbeddingRequest, EmbeddingResult]] = []

        def on_response(req: EmbeddingRequest, resp: EmbeddingResult) -> None:
            pairs.append((req, resp))

        client = EmbeddingClient(
            mock_lg, mock_backend, model="override"
        ).with_callbacks(EmbeddingCallbacks(on_response=on_response))
        client.embed("hi", context={"op": "x"})
        client.close()

        assert len(pairs) == 1
        req, resp = pairs[0]
        assert req.model == "override"
        assert req.context == {"op": "x"}
        assert resp is expected
        assert resp.prompt_tokens == 7

    def test_on_request_fires_before_backend_call(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """on_request fires with retry=0 before the backend runs."""
        mock_backend.embed.return_value = self._ok_result()
        events: list[tuple[str, int | None]] = []

        def on_request(_req: EmbeddingRequest, retry: int) -> None:
            events.append(("request", retry))

        mock_backend.embed.side_effect = lambda *a, **k: (
            events.append(("backend", None)) or self._ok_result()
        )

        client = EmbeddingClient(
            mock_lg, mock_backend, callbacks=EmbeddingCallbacks(on_request=on_request)
        )
        client.embed("hi")
        client.close()

        assert events == [("request", 0), ("backend", None)]

    def test_on_error_fires_on_terminal_failure(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """on_error fires with (request, exception) on non-retryable failure."""
        mock_backend.embed.side_effect = BackendRequestError(
            "bad request", status_code=400
        )
        captured: list[tuple[EmbeddingRequest, Exception]] = []

        def on_error(req: EmbeddingRequest, err: Exception) -> None:
            captured.append((req, err))

        client = EmbeddingClient(
            mock_lg, mock_backend, callbacks=EmbeddingCallbacks(on_error=on_error)
        )
        with pytest.raises(BackendRequestError):
            client.embed("hi", context={"op": "y"})
        client.close()

        assert len(captured) == 1
        req, err = captured[0]
        assert req.context == {"op": "y"}
        assert isinstance(err, BackendRequestError)
        assert err.status_code == 400

    def test_on_request_re_fires_per_retry(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """on_request fires again for each retry attempt, matching chat behavior."""
        retries: list[int] = []

        def on_request(_req: EmbeddingRequest, retry: int) -> None:
            retries.append(retry)

        call_count = 0

        def side_effect(text, *, model=None, dimensions=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise BackendRequestError("boom", status_code=503)
            return self._ok_result()

        mock_backend.embed.side_effect = side_effect

        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            retry=RetryConfig(base=0.001, factor=1.0, timeout=10.0),
            callbacks=EmbeddingCallbacks(on_request=on_request),
        )
        client.embed("hi")
        client.close()

        assert retries == [0, 1, 2]

    def test_on_error_fires_once_after_retries_exhausted(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """on_error fires exactly once when all retry attempts fail."""
        errors: list[Exception] = []

        def on_error(_req: EmbeddingRequest, err: Exception) -> None:
            errors.append(err)

        mock_backend.embed.side_effect = BackendRequestError("boom", status_code=503)

        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            retry=RetryConfig(base=0.001, factor=1.0, timeout=0.01),
            callbacks=EmbeddingCallbacks(on_error=on_error),
        )
        with pytest.raises(BackendRequestError):
            client.embed("hi")
        client.close()

        assert len(errors) == 1
        assert isinstance(errors[0], BackendRequestError)

    def test_on_retry_fires_with_raw_exception(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """on_retry fires once per transient retry with (req, exc, attempt, delay)."""
        retries: list[tuple[EmbeddingRequest, Exception, int, float]] = []

        def on_retry(
            req: EmbeddingRequest, err: Exception, attempt: int, delay: float
        ) -> None:
            retries.append((req, err, attempt, delay))

        transient = BackendRequestError("boom", status_code=429)
        call_count = 0

        def side_effect(text, *, model=None, dimensions=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise transient
            return self._ok_result()

        mock_backend.embed.side_effect = side_effect

        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            retry=RetryConfig(base=0.001, factor=1.0, timeout=10.0),
            callbacks=EmbeddingCallbacks(on_retry=on_retry),
        )
        client.embed("hi")
        client.close()

        assert len(retries) == 2
        # Raw exception object is preserved end to end.
        assert retries[0][1] is transient
        assert retries[0][2] == 1
        assert retries[0][3] > 0
        assert retries[1][2] == 2

    def test_on_retry_not_fired_on_terminal_error(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Non-transient failures skip on_retry, still fire on_error."""
        retries: list[tuple[EmbeddingRequest, Exception, int, float]] = []
        errors: list[Exception] = []

        mock_backend.embed.side_effect = BackendRequestError("bad", status_code=400)

        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            retry=RetryConfig(base=0.001, factor=1.0, timeout=10.0),
            callbacks=EmbeddingCallbacks(
                on_retry=lambda req, err, attempt, delay: retries.append(
                    (req, err, attempt, delay)
                ),
                on_error=lambda _req, err: errors.append(err),
            ),
        )
        with pytest.raises(BackendRequestError):
            client.embed("hi")
        client.close()

        assert retries == []
        assert len(errors) == 1

    def test_on_retry_callback_exception_swallowed(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """A raising on_retry does not derail the retry loop."""
        call_count = 0

        def side_effect(text, *, model=None, dimensions=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise BackendRequestError("boom", status_code=503)
            return self._ok_result()

        mock_backend.embed.side_effect = side_effect

        def bad_on_retry(*_: object) -> None:
            raise RuntimeError("tracker down")

        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            retry=RetryConfig(base=0.001, factor=1.0, timeout=10.0),
            callbacks=EmbeddingCallbacks(on_retry=bad_on_retry),
        )
        # Must NOT raise — the callback error is swallowed.
        client.embed("hi")
        client.close()

    @pytest.mark.asyncio
    async def test_on_retry_fires_async(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """on_retry fires on the async path with the raw exception."""
        retries: list[tuple[EmbeddingRequest, Exception, int, float]] = []
        transient = BackendUnavailableError("unavailable")
        call_count = 0

        def side_effect(text, *, model=None, dimensions=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise transient
            return self._ok_result()

        mock_backend.embed_async = AsyncMock(side_effect=side_effect)

        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            retry=RetryConfig(base=0.001, factor=1.0, timeout=10.0),
            callbacks=EmbeddingCallbacks(
                on_retry=lambda req, err, attempt, delay: retries.append(
                    (req, err, attempt, delay)
                )
            ),
        )
        await client.embed_async("hi")
        await client.aclose()

        assert len(retries) == 1
        assert retries[0][1] is transient
        assert retries[0][2] == 1

    def test_no_callbacks_preserves_original_behavior(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Without callbacks or context, embed still returns the backend result."""
        expected = self._ok_result()
        mock_backend.embed.return_value = expected

        client = EmbeddingClient(mock_lg, mock_backend)
        result = client.embed("hi")
        client.close()

        assert result is expected

    def test_callback_exception_does_not_kill_call(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """A raising callback is caught and logged; the call still returns."""
        expected = self._ok_result()
        mock_backend.embed.return_value = expected

        def on_response(_req: EmbeddingRequest, _resp: EmbeddingResult) -> None:
            raise RuntimeError("tracker down")

        client = EmbeddingClient(
            mock_lg, mock_backend, callbacks=EmbeddingCallbacks(on_response=on_response)
        )
        # Must NOT raise — callback isolation matches LLMClient.
        result = client.embed("hi")
        client.close()

        assert result is expected

    def test_with_callbacks_returns_clone(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """with_callbacks returns a new client; original is unaffected."""
        expected = self._ok_result()
        mock_backend.embed.return_value = expected
        seen: list[EmbeddingRequest] = []

        def on_response(req: EmbeddingRequest, _resp: EmbeddingResult) -> None:
            seen.append(req)

        base = EmbeddingClient(mock_lg, mock_backend)
        traced = base.with_callbacks(EmbeddingCallbacks(on_response=on_response))

        traced.embed("with", context={"op": "yes"})
        base.embed("without", context={"op": "no"})
        traced.close()

        assert len(seen) == 1
        assert seen[0].context == {"op": "yes"}

    def test_embed_batch_context_and_on_response(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """embed_batch delivers context on the request and on_response with the batch result."""
        expected = self._ok_batch(size=3, tokens=42)
        mock_backend.embed_batch.return_value = expected
        pairs: list[tuple[EmbeddingRequest, BatchEmbeddingResult]] = []

        def on_response(req: EmbeddingRequest, resp: BatchEmbeddingResult) -> None:
            pairs.append((req, resp))

        client = EmbeddingClient(
            mock_lg, mock_backend, callbacks=EmbeddingCallbacks(on_response=on_response)
        )
        client.embed_batch(["a", "b", "c"], context={"op": "batch"})
        client.close()

        assert len(pairs) == 1
        req, resp = pairs[0]
        assert req.context == {"op": "batch"}
        assert req.texts == ["a", "b", "c"]
        assert resp is expected
        assert resp.total_prompt_tokens == 42

    def test_embed_batch_empty_skips_callbacks(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Empty batch short-circuits: no backend call, no callback fire."""
        fired: list[str] = []
        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            callbacks=EmbeddingCallbacks(
                on_request=lambda _r, _n: fired.append("request"),
                on_response=lambda _r, _x: fired.append("response"),
                on_error=lambda _r, _e: fired.append("error"),
            ),
        )
        result = client.embed_batch([])
        client.close()

        assert result.embeddings == []
        assert fired == []
        mock_backend.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_async_context_and_on_response(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """embed_async delivers context and fires on_response with the async result."""
        expected = self._ok_result()

        async def _embed_async(text, *, model=None, dimensions=None):
            return expected

        mock_backend.embed_async = _embed_async
        pairs: list[tuple[EmbeddingRequest, EmbeddingResult]] = []

        def on_response(req: EmbeddingRequest, resp: EmbeddingResult) -> None:
            pairs.append((req, resp))

        client = EmbeddingClient(
            mock_lg, mock_backend, callbacks=EmbeddingCallbacks(on_response=on_response)
        )
        await client.embed_async("hi", context={"op": "async"})
        await client.aclose()

        assert len(pairs) == 1
        assert pairs[0][0].context == {"op": "async"}
        assert pairs[0][1] is expected

    @pytest.mark.asyncio
    async def test_embed_async_on_error(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """embed_async fires on_error on terminal failure."""

        async def _embed_async_fail(text, *, model=None, dimensions=None):
            raise BackendRequestError("nope", status_code=400)

        mock_backend.embed_async = _embed_async_fail
        captured: list[Exception] = []

        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            callbacks=EmbeddingCallbacks(on_error=lambda _r, e: captured.append(e)),
        )
        with pytest.raises(BackendRequestError):
            await client.embed_async("hi")
        await client.aclose()

        assert len(captured) == 1
        assert isinstance(captured[0], BackendRequestError)

    @pytest.mark.asyncio
    async def test_embed_batch_async_context(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """embed_batch_async threads context onto the batch request."""
        expected = self._ok_batch(size=2, tokens=17)

        async def _embed_batch_async(texts, *, model=None, dimensions=None):
            return expected

        mock_backend.embed_batch_async = _embed_batch_async
        captured: list[EmbeddingRequest] = []

        client = EmbeddingClient(
            mock_lg,
            mock_backend,
            callbacks=EmbeddingCallbacks(
                on_response=lambda req, _r: captured.append(req)
            ),
        )
        await client.embed_batch_async(["a", "b"], context={"op": "abatch"})
        await client.aclose()

        assert len(captured) == 1
        assert captured[0].context == {"op": "abatch"}
        assert captured[0].texts == ["a", "b"]


class TestEmbeddingClientFactory:
    """Test factory methods for creating EmbeddingClient."""

    def test_factory_embeddings_creates_openai_backend(self, mock_lg: Logger) -> None:
        """Test Factory.embeddings() creates client with OpenAI backend."""
        from llm_infer.client import Factory

        factory = Factory(mock_lg)
        client = factory.embeddings(
            base_url="http://localhost:8001/v1",
            model="text-embedding-3-small",
            api_key="test-key",
        )

        assert isinstance(client, EmbeddingClient)
        assert isinstance(client.backend, OpenAIBackend)
        assert client.model == "text-embedding-3-small"
        client.close()

    def test_factory_embeddings_google_creates_google_backend(
        self, mock_lg: Logger
    ) -> None:
        """Test Factory.embeddings_google() creates client with Google backend."""
        from llm_infer.client import Factory
        from llm_infer.client.backends.embedding import GoogleBackend

        factory = Factory(mock_lg)
        client = factory.embeddings_google(
            api_key="test-key",
            model="text-embedding-004",
            task_type="RETRIEVAL_DOCUMENT",
        )

        assert isinstance(client, EmbeddingClient)
        assert isinstance(client.backend, GoogleBackend)
        assert client.model == "text-embedding-004"
        client.close()

    def test_factory_embeddings_passes_callbacks(self, mock_lg: Logger) -> None:
        """Factory.embeddings forwards callbacks= to the client."""
        from llm_infer.client import Factory

        cbs = EmbeddingCallbacks(on_response=lambda _r, _x: None)
        factory = Factory(mock_lg)
        client = factory.embeddings(
            base_url="http://localhost:8001/v1",
            model="m",
            api_key="k",
            callbacks=cbs,
        )
        assert client._callbacks is cbs
        client.close()

    def test_factory_embeddings_from_config_passes_callbacks(
        self, mock_lg: Logger
    ) -> None:
        """Factory.embeddings_from_config forwards callbacks= to the client."""
        from llm_infer.client import Factory

        cbs = EmbeddingCallbacks(on_response=lambda _r, _x: None)
        factory = Factory(mock_lg)
        client = factory.embeddings_from_config(
            {"type": "openai", "base_url": "http://localhost:8001/v1", "model": "m"},
            callbacks=cbs,
        )
        assert client._callbacks is cbs
        client.close()
