"""Unit tests for OpenAI embedding backend."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from appinfra.log import Logger

from llm_infer.client import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_infer.client.backends import BackendContext
from llm_infer.client.backends.embedding import EmbeddingResult
from llm_infer.client.backends.providers import OpenAIEmbeddingBackend

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


class TestOpenAIEmbeddingBackendInit:
    """Test OpenAIEmbeddingBackend initialization."""

    def test_defaults(self, mock_lg: Logger) -> None:
        """Test default configuration."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="text-embedding-3-small"
        )
        assert backend.model == "text-embedding-3-small"
        assert backend.provider == "openai"
        backend.close()

    def test_strips_trailing_slash(self, mock_lg: Logger) -> None:
        """Test trailing slash is stripped from base_url."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1/", model="model"
        )
        assert backend._base_url == "http://localhost:8001/v1"
        backend.close()

    def test_api_key_sets_auth_header(self, mock_lg: Logger) -> None:
        """Test API key is used in Authorization header."""
        backend = OpenAIEmbeddingBackend(
            mock_lg,
            base_url="http://localhost:8001/v1",
            model="model",
            api_key="sk-test123",
        )
        assert backend._headers["Authorization"] == "Bearer sk-test123"
        backend.close()


class TestOpenAIEmbeddingBackendEmbed:
    """Test embed method."""

    def test_embed_single_text(self, mock_lg: Logger) -> None:
        """Test embedding a single text."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

        embedding = [0.1, 0.2, 0.3]
        response = make_embedding_response([embedding], prompt_tokens=5)

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = backend.embed("hello world")

        assert isinstance(result, EmbeddingResult)
        assert result.embedding == embedding
        assert result.model == "text-embedding-model"
        assert result.dimensions == 3
        assert result.prompt_tokens == 5

        mock_post.assert_called_once_with(
            "http://localhost:8001/v1/embeddings",
            json={"model": "default", "input": "hello world"},
        )
        backend.close()


class TestOpenAIEmbeddingBackendEmbedBatch:
    """Test embed_batch method."""

    def test_embed_batch_multiple_texts(self, mock_lg: Logger) -> None:
        """Test embedding multiple texts."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        response = make_embedding_response(embeddings, prompt_tokens=15)

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = backend.embed_batch(["a", "b", "c"])

        assert len(result.results) == 3
        assert result.results[0].embedding == [0.1, 0.2]
        assert result.results[1].embedding == [0.3, 0.4]
        assert result.results[2].embedding == [0.5, 0.6]
        assert result.total_prompt_tokens == 15
        assert all(r.prompt_tokens is None for r in result.results)

        mock_post.assert_called_once_with(
            "http://localhost:8001/v1/embeddings",
            json={"model": "default", "input": ["a", "b", "c"]},
        )
        backend.close()

    def test_embed_batch_empty_list(self, mock_lg: Logger) -> None:
        """Test embedding empty list returns empty result without API call."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

        with patch.object(backend._client, "post") as mock_post:
            result = backend.embed_batch([])

        assert result.results == []
        assert result.total_prompt_tokens == 0
        mock_post.assert_not_called()
        backend.close()

    def test_embed_batch_preserves_order(self, mock_lg: Logger) -> None:
        """Test batch results maintain input order."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

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

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = backend.embed_batch(["a", "b", "c"])

        assert result.results[0].embedding == [0.1]
        assert result.results[1].embedding == [0.2]
        assert result.results[2].embedding == [0.3]
        backend.close()


class TestOpenAIEmbeddingBackendErrorHandling:
    """Test error translation."""

    def test_connection_error_raises_unavailable(self, mock_lg: Logger) -> None:
        """Test connection error is translated to BackendUnavailableError."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

        with patch.object(backend._client, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(BackendUnavailableError, match="Failed to connect"):
                backend.embed("test")

        backend.close()

    def test_timeout_error_raises_timeout(self, mock_lg: Logger) -> None:
        """Test timeout is translated to BackendTimeoutError."""
        backend = OpenAIEmbeddingBackend(
            mock_lg,
            base_url="http://localhost:8001/v1",
            model="default",
            ctx=BackendContext(request_timeout=5.0),
        )

        with patch.object(backend._client, "post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Timed out")

            with pytest.raises(BackendTimeoutError, match="timed out"):
                backend.embed("test")

        backend.close()

    def test_http_error_raises_request_error(self, mock_lg: Logger) -> None:
        """Test HTTP error is translated to BackendRequestError with status code."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            mock_post.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_response
            )

            with pytest.raises(BackendRequestError) as exc_info:
                backend.embed("test")

            assert exc_info.value.status_code == 400

        backend.close()

    def test_json_decode_error_raises_request_error(self, mock_lg: Logger) -> None:
        """Test JSON decode error is translated to BackendRequestError."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
            mock_post.return_value = mock_response

            with pytest.raises(BackendRequestError, match="Invalid JSON"):
                backend.embed("test")

        backend.close()


class TestOpenAIEmbeddingBackendAsync:
    """Test async methods."""

    @pytest.mark.asyncio
    async def test_embed_async(self, mock_lg: Logger) -> None:
        """Test async embedding."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

        response = make_embedding_response([[0.1, 0.2]])

        async_client = backend._get_async_client()
        with patch.object(async_client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()

            async def async_post(*args, **kwargs):
                return mock_response

            mock_post.side_effect = async_post

            result = await backend.embed_async("test")

        assert result.embedding == [0.1, 0.2]
        await backend.aclose()

    @pytest.mark.asyncio
    async def test_embed_batch_async_empty(self, mock_lg: Logger) -> None:
        """Test async batch with empty list."""
        backend = OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        )

        result = await backend.embed_batch_async([])

        assert result.results == []
        assert result.total_prompt_tokens == 0
        await backend.aclose()


class TestOpenAIEmbeddingBackendContextManager:
    """Test context manager support."""

    def test_sync_context_manager(self, mock_lg: Logger) -> None:
        """Test sync context manager."""
        with OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        ) as backend:
            assert isinstance(backend, OpenAIEmbeddingBackend)
        assert backend._client.is_closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_lg: Logger) -> None:
        """Test async context manager."""
        async with OpenAIEmbeddingBackend(
            mock_lg, base_url="http://localhost:8001/v1", model="default"
        ) as backend:
            assert isinstance(backend, OpenAIEmbeddingBackend)
        assert backend._client.is_closed
