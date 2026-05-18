"""Unit tests for EmbeddingsClient."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from appinfra.log import Logger

from llm_infer.client import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    EmbeddingResult,
    EmbeddingsClient,
    RetryConfig,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


def make_embedding_response(
    embeddings: list[list[float]],
    model: str = "text-embedding-model",
    prompt_tokens: int = 10,
) -> dict:
    """Create a mock embedding response."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings)
        ],
        "model": model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


class TestEmbeddingsClientInit:
    """Test EmbeddingsClient initialization."""

    def test_defaults(self, mock_lg: Logger) -> None:
        """Test default configuration."""
        client = EmbeddingsClient(mock_lg)
        assert client._base_url == "http://localhost:8001/v1"
        assert client._request_model == "default"
        assert client._timeout == 30.0
        assert client._retry is None
        client.close()

    def test_custom_configuration(self, mock_lg: Logger) -> None:
        """Test custom configuration."""
        retry = RetryConfig(base=1.0, factor=2.0, timeout=60.0)
        client = EmbeddingsClient(
            mock_lg,
            base_url="http://localhost:8000/v1/",  # trailing slash should be stripped
            model="custom-model",
            timeout=60.0,
            retry=retry,
        )
        assert client._base_url == "http://localhost:8000/v1"
        assert client._request_model == "custom-model"
        assert client._timeout == 60.0
        assert client._retry is retry
        client.close()


class TestEmbeddingsClientModel:
    """Test model discovery and caching."""

    def test_model_returns_request_model_before_discovery(
        self, mock_lg: Logger
    ) -> None:
        """Test model property returns request model before any requests."""
        client = EmbeddingsClient(mock_lg, model="my-model")
        assert client.model == "my-model"
        client.close()

    def test_model_returns_discovered_model(self, mock_lg: Logger) -> None:
        """Test model property returns discovered model after request."""
        client = EmbeddingsClient(mock_lg, model="requested")

        response = make_embedding_response([[0.1, 0.2]], model="actual-model")

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            client.embed("test")

        assert client.model == "actual-model"
        client.close()


class TestEmbeddingsClientEmbed:
    """Test embed method."""

    def test_embed_single_text(self, mock_lg: Logger) -> None:
        """Test embedding a single text."""
        client = EmbeddingsClient(mock_lg)

        embedding = [0.1, 0.2, 0.3]
        response = make_embedding_response([embedding], prompt_tokens=5)

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = client.embed("hello world")

        assert isinstance(result, EmbeddingResult)
        assert result.embedding == embedding
        assert result.model == "text-embedding-model"
        assert result.prompt_tokens == 5

        mock_post.assert_called_once_with(
            "http://localhost:8001/v1/embeddings",
            json={"model": "default", "input": "hello world"},
        )
        client.close()

    def test_embed_empty_text(self, mock_lg: Logger) -> None:
        """Test embedding empty text (model discovery probe)."""
        client = EmbeddingsClient(mock_lg)

        response = make_embedding_response([[]], prompt_tokens=0)

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = client.embed("")

        assert result.embedding == []
        mock_post.assert_called_once()
        client.close()


class TestEmbeddingsClientEmbedBatch:
    """Test embed_batch method."""

    def test_embed_batch_multiple_texts(self, mock_lg: Logger) -> None:
        """Test embedding multiple texts."""
        client = EmbeddingsClient(mock_lg)

        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        response = make_embedding_response(embeddings, prompt_tokens=15)

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            results = client.embed_batch(["a", "b", "c"])

        assert len(results) == 3
        assert results[0].embedding == [0.1, 0.2]
        assert results[1].embedding == [0.3, 0.4]
        assert results[2].embedding == [0.5, 0.6]
        # 15 tokens / 3 texts = 5 per text
        assert all(r.prompt_tokens == 5 for r in results)

        mock_post.assert_called_once_with(
            "http://localhost:8001/v1/embeddings",
            json={"model": "default", "input": ["a", "b", "c"]},
        )
        client.close()

    def test_embed_batch_empty_list(self, mock_lg: Logger) -> None:
        """Test embedding empty list returns empty list without API call."""
        client = EmbeddingsClient(mock_lg)

        with patch.object(client._client, "post") as mock_post:
            results = client.embed_batch([])

        assert results == []
        mock_post.assert_not_called()
        client.close()

    def test_embed_batch_preserves_order(self, mock_lg: Logger) -> None:
        """Test batch results maintain input order even if server returns different order."""
        client = EmbeddingsClient(mock_lg)

        # Server returns out of order
        response = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.3], "index": 2},
                {"object": "embedding", "embedding": [0.1], "index": 0},
                {"object": "embedding", "embedding": [0.2], "index": 1},
            ],
            "model": "model",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            results = client.embed_batch(["a", "b", "c"])

        # Results should be sorted by index
        assert results[0].embedding == [0.1]
        assert results[1].embedding == [0.2]
        assert results[2].embedding == [0.3]
        client.close()


class TestEmbeddingsClientErrorHandling:
    """Test error translation."""

    def test_connection_error_raises_unavailable(self, mock_lg: Logger) -> None:
        """Test connection error is translated to BackendUnavailableError."""
        client = EmbeddingsClient(mock_lg)

        with patch.object(client._client, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(BackendUnavailableError, match="Failed to connect"):
                client.embed("test")

        client.close()

    def test_timeout_error_raises_timeout(self, mock_lg: Logger) -> None:
        """Test timeout is translated to BackendTimeoutError."""
        client = EmbeddingsClient(mock_lg, timeout=5.0)

        with patch.object(client._client, "post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Timed out")

            with pytest.raises(BackendTimeoutError, match="timed out"):
                client.embed("test")

        client.close()

    def test_http_error_raises_request_error(self, mock_lg: Logger) -> None:
        """Test HTTP error is translated to BackendRequestError with status code."""
        client = EmbeddingsClient(mock_lg)

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            mock_post.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_response
            )

            with pytest.raises(BackendRequestError) as exc_info:
                client.embed("test")

            assert exc_info.value.status_code == 400

        client.close()

    def test_json_decode_error_raises_request_error(self, mock_lg: Logger) -> None:
        """Test JSON decode error is translated to BackendRequestError."""
        client = EmbeddingsClient(mock_lg)

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
            mock_post.return_value = mock_response

            with pytest.raises(BackendRequestError, match="Invalid JSON"):
                client.embed("test")

        client.close()


class TestEmbeddingsClientRetry:
    """Test retry behavior."""

    def test_retry_on_5xx_error(self, mock_lg: Logger) -> None:
        """Test retry on 5xx server error."""
        retry = RetryConfig(base=0.01, factor=1.0, timeout=10.0)
        client = EmbeddingsClient(mock_lg, retry=retry)

        response_ok = make_embedding_response([[0.1]])

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                mock_response = MagicMock()
                mock_response.status_code = 503
                mock_response.text = "Service unavailable"
                raise httpx.HTTPStatusError(
                    "Error", request=MagicMock(), response=mock_response
                )
            mock_response = MagicMock()
            mock_response.json.return_value = response_ok
            mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch.object(client._client, "post") as mock_post:
            mock_post.side_effect = side_effect

            result = client.embed("test")

        assert call_count == 3
        assert result.embedding == [0.1]
        client.close()

    def test_retry_on_429_rate_limit(self, mock_lg: Logger) -> None:
        """Test retry on 429 rate limit."""
        retry = RetryConfig(base=0.01, factor=1.0, timeout=10.0)
        client = EmbeddingsClient(mock_lg, retry=retry)

        response_ok = make_embedding_response([[0.1]])

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock_response = MagicMock()
                mock_response.status_code = 429
                mock_response.text = "Rate limited"
                raise httpx.HTTPStatusError(
                    "Error", request=MagicMock(), response=mock_response
                )
            mock_response = MagicMock()
            mock_response.json.return_value = response_ok
            mock_response.raise_for_status = MagicMock()
            return mock_response

        with patch.object(client._client, "post") as mock_post:
            mock_post.side_effect = side_effect

            client.embed("test")

        assert call_count == 2
        client.close()

    def test_no_retry_on_4xx_client_error(self, mock_lg: Logger) -> None:
        """Test no retry on 4xx client errors (except 429, 529)."""
        retry = RetryConfig(base=0.01, factor=1.0, timeout=10.0)
        client = EmbeddingsClient(mock_lg, retry=retry)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            raise httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_response
            )

        with patch.object(client._client, "post") as mock_post:
            mock_post.side_effect = side_effect

            with pytest.raises(BackendRequestError) as exc_info:
                client.embed("test")

            assert exc_info.value.status_code == 400

        # Should not retry on 400
        assert call_count == 1
        client.close()

    def test_no_retry_when_disabled(self, mock_lg: Logger) -> None:
        """Test no retry when retry config is None."""
        client = EmbeddingsClient(mock_lg, retry=None)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.text = "Service unavailable"
            raise httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_response
            )

        with patch.object(client._client, "post") as mock_post:
            mock_post.side_effect = side_effect

            with pytest.raises(BackendRequestError):
                client.embed("test")

        assert call_count == 1
        client.close()


class TestEmbeddingsClientAsync:
    """Test async methods."""

    @pytest.mark.asyncio
    async def test_embed_async(self, mock_lg: Logger) -> None:
        """Test async embedding."""
        client = EmbeddingsClient(mock_lg)

        response = make_embedding_response([[0.1, 0.2]])

        async_client = client._get_async_client()
        with patch.object(async_client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()

            # Create an awaitable that returns the mock
            async def async_post(*args, **kwargs):
                return mock_response

            mock_post.side_effect = async_post

            result = await client.embed_async("test")

        assert result.embedding == [0.1, 0.2]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_embed_batch_async_empty(self, mock_lg: Logger) -> None:
        """Test async batch with empty list."""
        client = EmbeddingsClient(mock_lg)

        results = await client.embed_batch_async([])

        assert results == []
        await client.aclose()


class TestEmbeddingsClientDiscovery:
    """Test model discovery methods."""

    def test_discover_makes_probe_request(self, mock_lg: Logger) -> None:
        """Test discover() makes a probe request and returns model."""
        client = EmbeddingsClient(mock_lg, model="requested")

        response = make_embedding_response([[]], model="actual-model")

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            model = client.discover()

        assert model == "actual-model"
        # Probe with empty string
        mock_post.assert_called_once_with(
            "http://localhost:8001/v1/embeddings",
            json={"model": "requested", "input": ""},
        )
        client.close()

    def test_discover_caches_result(self, mock_lg: Logger) -> None:
        """Test discover() only probes once."""
        client = EmbeddingsClient(mock_lg)

        response = make_embedding_response([[]], model="discovered")

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            model1 = client.discover()
            model2 = client.discover()

        assert model1 == model2 == "discovered"
        assert mock_post.call_count == 1
        client.close()


class TestEmbeddingsClientContextManager:
    """Test context manager support."""

    def test_sync_context_manager(self, mock_lg: Logger) -> None:
        """Test sync context manager."""
        with EmbeddingsClient(mock_lg) as client:
            assert isinstance(client, EmbeddingsClient)
        # Client should be closed after context exits
        assert client._client.is_closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_lg: Logger) -> None:
        """Test async context manager."""
        async with EmbeddingsClient(mock_lg) as client:
            assert isinstance(client, EmbeddingsClient)
        # Sync client should be closed
        assert client._client.is_closed

    def test_close_is_idempotent(self, mock_lg: Logger) -> None:
        """Test close() can be called multiple times."""
        client = EmbeddingsClient(mock_lg)
        client.close()
        client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self, mock_lg: Logger) -> None:
        """Test aclose() can be called multiple times."""
        client = EmbeddingsClient(mock_lg)
        await client.aclose()
        await client.aclose()  # Should not raise
