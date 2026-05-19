"""Unit tests for EmbeddingClient."""

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.client import (
    BackendRequestError,
    BackendUnavailableError,
    EmbeddingClient,
    EmbeddingResult,
    RetryConfig,
)
from llm_infer.client.backends.embedding import Backend, OpenAIBackend

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
        mock_backend.embed.assert_called_once_with("hello", dimensions=None)
        client.close()

    def test_embed_batch_delegates_to_backend(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test embed_batch delegates to backend."""
        expected = [
            EmbeddingResult(
                embedding=[0.1], model="model", dimensions=1, prompt_tokens=5
            ),
            EmbeddingResult(
                embedding=[0.2], model="model", dimensions=1, prompt_tokens=5
            ),
        ]
        mock_backend.embed_batch.return_value = expected

        client = EmbeddingClient(mock_lg, mock_backend)
        results = client.embed_batch(["a", "b"])

        assert results is expected
        mock_backend.embed_batch.assert_called_once_with(["a", "b"], dimensions=None)
        client.close()

    def test_embed_batch_empty_list(
        self, mock_lg: Logger, mock_backend: MagicMock
    ) -> None:
        """Test embed_batch with empty list doesn't call backend."""
        client = EmbeddingClient(mock_lg, mock_backend)
        results = client.embed_batch([])

        assert results == []
        mock_backend.embed_batch.assert_not_called()
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

        def side_effect(text, *, dimensions=None):
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

        def side_effect(text, *, dimensions=None):
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

        async def mock_embed_async(text, *, dimensions=None):
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
        results = await client.embed_batch_async([])

        assert results == []
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

        async def mock_aclose():
            pass

        mock_backend.aclose = mock_aclose

        async with EmbeddingClient(mock_lg, mock_backend) as client:
            assert isinstance(client, EmbeddingClient)


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
