"""Unit tests for Google embedding backend."""

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
from llm_infer.client.backends.providers import (
    GoogleEmbeddingBackend,
    GoogleEmbeddingTaskType,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_lg() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


def make_google_single_response(embedding: list[float]) -> dict:
    """Create a mock Google embedContent response."""
    return {"embedding": {"values": embedding}}


def make_google_batch_response(embeddings: list[list[float]]) -> dict:
    """Create a mock Google batchEmbedContents response."""
    return {"embeddings": [{"values": emb} for emb in embeddings]}


class TestGoogleEmbeddingBackendInit:
    """Test GoogleEmbeddingBackend initialization."""

    def test_defaults(self, mock_lg: Logger) -> None:
        """Test default configuration."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")
        assert backend.model == "gemini-embedding-001"
        assert backend.task_type == GoogleEmbeddingTaskType.RETRIEVAL_DOCUMENT
        assert backend.provider == "google"
        backend.close()

    def test_custom_configuration(self, mock_lg: Logger) -> None:
        """Test custom configuration."""
        backend = GoogleEmbeddingBackend(
            mock_lg,
            api_key="test-key",
            model="custom-model",
            task_type=GoogleEmbeddingTaskType.RETRIEVAL_QUERY,
            ctx=BackendContext(request_timeout=60.0),
        )
        assert backend.model == "custom-model"
        assert backend.task_type == GoogleEmbeddingTaskType.RETRIEVAL_QUERY
        assert backend._ctx.request_timeout == 60.0
        backend.close()

    def test_api_key_sets_header(self, mock_lg: Logger) -> None:
        """Test API key is used in x-goog-api-key header."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="my-api-key")
        assert backend._build_headers()["x-goog-api-key"] == "my-api-key"
        backend.close()


class TestGoogleEmbeddingTaskType:
    """Test GoogleEmbeddingTaskType enum."""

    def test_all_task_types(self) -> None:
        """Test all task types are valid."""
        assert GoogleEmbeddingTaskType.RETRIEVAL_QUERY == "RETRIEVAL_QUERY"
        assert GoogleEmbeddingTaskType.RETRIEVAL_DOCUMENT == "RETRIEVAL_DOCUMENT"
        assert GoogleEmbeddingTaskType.SEMANTIC_SIMILARITY == "SEMANTIC_SIMILARITY"
        assert GoogleEmbeddingTaskType.CLASSIFICATION == "CLASSIFICATION"
        assert GoogleEmbeddingTaskType.CLUSTERING == "CLUSTERING"
        assert GoogleEmbeddingTaskType.QUESTION_ANSWERING == "QUESTION_ANSWERING"
        assert GoogleEmbeddingTaskType.FACT_VERIFICATION == "FACT_VERIFICATION"


class TestGoogleEmbeddingBackendEmbed:
    """Test embed method."""

    def test_embed_single_text(self, mock_lg: Logger) -> None:
        """Test embedding a single text."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        embedding = [0.1, 0.2, 0.3]
        response = make_google_single_response(embedding)

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = backend.embed("hello world")

        assert isinstance(result, EmbeddingResult)
        assert result.embedding == embedding
        assert result.model == "gemini-embedding-001"
        assert result.dimensions == 3
        assert result.prompt_tokens is None  # Google doesn't return token counts

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        call_json = mock_post.call_args[1]["json"]

        assert "models/gemini-embedding-001:embedContent" in call_url
        assert call_json["content"]["parts"][0]["text"] == "hello world"
        assert call_json["taskType"] == "RETRIEVAL_DOCUMENT"
        backend.close()

    def test_embed_with_different_task_type(self, mock_lg: Logger) -> None:
        """Test embedding with a different task type."""
        backend = GoogleEmbeddingBackend(
            mock_lg,
            api_key="test-key",
            task_type=GoogleEmbeddingTaskType.RETRIEVAL_QUERY,
        )

        response = make_google_single_response([0.1])

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            backend.embed("query text")

        call_json = mock_post.call_args[1]["json"]
        assert call_json["taskType"] == "RETRIEVAL_QUERY"
        backend.close()


class TestGoogleEmbeddingBackendEmbedBatch:
    """Test embed_batch method."""

    def test_embed_batch_multiple_texts(self, mock_lg: Logger) -> None:
        """Test embedding multiple texts."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        response = make_google_batch_response(embeddings)

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = response
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            result = backend.embed_batch(["a", "b", "c"])

        assert len(result.embeddings) == 3
        assert result.embeddings[0] == [0.1, 0.2]
        assert result.embeddings[1] == [0.3, 0.4]
        assert result.embeddings[2] == [0.5, 0.6]
        assert result.model == "gemini-embedding-001"
        assert result.dimensions == 2
        assert result.size == 3
        assert result.total_prompt_tokens is None  # Google API doesn't return tokens

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        call_json = mock_post.call_args[1]["json"]

        assert "batchEmbedContents" in call_url
        assert len(call_json["requests"]) == 3
        backend.close()

    def test_embed_batch_empty_list(self, mock_lg: Logger) -> None:
        """Test embedding empty list returns empty result without API call."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        with patch.object(backend._client, "post") as mock_post:
            result = backend.embed_batch([])

        assert result.embeddings == []
        assert result.total_prompt_tokens is None
        mock_post.assert_not_called()
        backend.close()

    def test_embed_batch_exceeds_max_size(self, mock_lg: Logger) -> None:
        """Test batch exceeding max size raises error."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        texts = ["text"] * (backend.max_batch_size + 1)

        with pytest.raises(BackendRequestError, match="exceeds maximum"):
            backend.embed_batch(texts)

        backend.close()


class TestGoogleEmbeddingBackendErrorHandling:
    """Test error translation."""

    def test_connection_error_raises_unavailable(self, mock_lg: Logger) -> None:
        """Test connection error is translated to BackendUnavailableError."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        with patch.object(backend._client, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(BackendUnavailableError, match="Failed to connect"):
                backend.embed("test")

        backend.close()

    def test_timeout_error_raises_timeout(self, mock_lg: Logger) -> None:
        """Test timeout is translated to BackendTimeoutError."""
        backend = GoogleEmbeddingBackend(
            mock_lg, api_key="test-key", ctx=BackendContext(request_timeout=5.0)
        )

        with patch.object(backend._client, "post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Timed out")

            with pytest.raises(BackendTimeoutError, match="timed out"):
                backend.embed("test")

        backend.close()

    def test_http_error_raises_request_error(self, mock_lg: Logger) -> None:
        """Test HTTP error is translated to BackendRequestError with status code."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

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
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        with patch.object(backend._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
            mock_post.return_value = mock_response

            with pytest.raises(BackendRequestError, match="Invalid JSON"):
                backend.embed("test")

        backend.close()


class TestGoogleEmbeddingBackendAsync:
    """Test async methods."""

    @pytest.mark.asyncio
    async def test_embed_async(self, mock_lg: Logger) -> None:
        """Test async embedding."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        response = make_google_single_response([0.1, 0.2])

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
        assert async_client.is_closed

    @pytest.mark.asyncio
    async def test_embed_batch_async_empty(self, mock_lg: Logger) -> None:
        """Test async batch with empty list."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        result = await backend.embed_batch_async([])

        assert result.embeddings == []
        assert result.total_prompt_tokens is None
        await backend.aclose()

    @pytest.mark.asyncio
    async def test_embed_batch_async_exceeds_max_size(self, mock_lg: Logger) -> None:
        """Test async batch exceeding max size raises error."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")

        texts = ["text"] * (backend.max_batch_size + 1)

        with pytest.raises(BackendRequestError, match="exceeds maximum"):
            await backend.embed_batch_async(texts)

        await backend.aclose()


class TestGoogleEmbeddingBackendContextManager:
    """Test context manager support."""

    def test_sync_context_manager(self, mock_lg: Logger) -> None:
        """Test sync context manager."""
        with GoogleEmbeddingBackend(mock_lg, api_key="test-key") as backend:
            assert isinstance(backend, GoogleEmbeddingBackend)
        assert backend._client.is_closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_lg: Logger) -> None:
        """Test async context manager."""
        async with GoogleEmbeddingBackend(mock_lg, api_key="test-key") as backend:
            assert isinstance(backend, GoogleEmbeddingBackend)
            async_client = backend._get_async_client()
        assert backend._client.is_closed
        assert async_client.is_closed


class TestGoogleEmbeddingBackendVertex:
    """Test Vertex AI configuration: base_url override + SA auth."""

    def test_requires_api_key_or_auth(self, mock_lg: Logger) -> None:
        """Constructor rejects when neither api_key nor auth is provided."""
        from pathlib import Path

        with pytest.raises(ValueError, match="api_key or auth"):
            GoogleEmbeddingBackend(mock_lg)
        # Ensure the path doesn't matter — pure validation.
        assert isinstance(Path, type)

    def test_base_url_override(self, mock_lg: Logger) -> None:
        """Custom base_url is used for request URLs."""
        backend = GoogleEmbeddingBackend(
            mock_lg,
            api_key="test-key",
            base_url="https://aiplatform.googleapis.com/v1",
        )
        assert backend._base_url == "https://aiplatform.googleapis.com/v1"
        backend.close()

    def test_default_base_url_unchanged(self, mock_lg: Logger) -> None:
        """Default base_url is AI Studio for backwards compat."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="test-key")
        assert backend._base_url == ("https://generativelanguage.googleapis.com/v1beta")
        backend.close()

    def test_base_url_strips_trailing_slash(self, mock_lg: Logger) -> None:
        backend = GoogleEmbeddingBackend(
            mock_lg, api_key="k", base_url="https://example.com/v1/"
        )
        assert backend._base_url == "https://example.com/v1"
        backend.close()

    def test_request_url_uses_overridden_base_url(self, mock_lg: Logger) -> None:
        """The :embedContent path is built off the overridden base_url."""
        backend = GoogleEmbeddingBackend(
            mock_lg,
            api_key="k",
            base_url="https://aiplatform.googleapis.com/v1",
            model="google/gemini-embedding-001",
        )
        captured: dict = {}

        def fake_post(url: str, json: dict, headers: dict) -> object:
            captured["url"] = url
            captured["headers"] = headers
            resp = MagicMock()
            resp.json.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
            resp.raise_for_status = MagicMock()
            return resp

        with patch.object(backend._client, "post", side_effect=fake_post):
            backend.embed("hello")

        assert captured["url"] == (
            "https://aiplatform.googleapis.com/v1/models/"
            "google/gemini-embedding-001:embedContent"
        )
        assert captured["headers"]["x-goog-api-key"] == "k"
        backend.close()

    def test_vertex_omits_model_from_single_body(self, mock_lg: Logger) -> None:
        """Vertex rejects body 'model' (URL already has it); omit on Vertex."""
        backend = GoogleEmbeddingBackend(
            mock_lg,
            api_key="k",
            base_url="https://us-central1-aiplatform.googleapis.com/v1",
            model="gemini-embedding-001",
        )
        payload = backend._build_single_request("hi")
        assert "model" not in payload
        assert payload["content"]["parts"][0]["text"] == "hi"
        backend.close()

    def test_vertex_omits_model_from_batch_body(self, mock_lg: Logger) -> None:
        backend = GoogleEmbeddingBackend(
            mock_lg,
            api_key="k",
            base_url="https://aiplatform.googleapis.com/v1",
            model="gemini-embedding-001",
        )
        payload = backend._build_batch_request(["a", "b"])
        for r in payload["requests"]:
            assert "model" not in r
        backend.close()

    def test_ai_studio_keeps_model_in_body(self, mock_lg: Logger) -> None:
        """Regression guard: AI Studio body still carries 'model'."""
        backend = GoogleEmbeddingBackend(mock_lg, api_key="k")  # default URL
        payload = backend._build_single_request("hi")
        assert payload["model"] == "models/gemini-embedding-001"
        backend.close()

    def test_sa_auth_sends_bearer_header(self, mock_lg: Logger, tmp_path) -> None:
        """With GCP SA auth, Authorization: Bearer is sent instead of x-goog-api-key."""
        import json as _json

        from llm_infer.client.backends.auth import GCPServiceAccountAuth

        sa_file = tmp_path / "sa.json"
        sa_file.write_text(_json.dumps({"type": "service_account"}))

        mock_creds = MagicMock()
        mock_creds.token = "vertex-token"
        mock_creds.expiry = None

        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ):
            auth = GCPServiceAccountAuth(mock_lg, credentials_path=str(sa_file))
            backend = GoogleEmbeddingBackend(
                mock_lg,
                auth=auth,
                base_url="https://aiplatform.googleapis.com/v1",
                model="google/gemini-embedding-001",
            )

            captured: dict = {}

            def fake_post(url: str, json: dict, headers: dict) -> object:
                captured["headers"] = headers
                resp = MagicMock()
                resp.json.return_value = {"embedding": {"values": [0.1]}}
                resp.raise_for_status = MagicMock()
                return resp

            with patch.object(backend._client, "post", side_effect=fake_post):
                backend.embed("hello")

            assert captured["headers"]["Authorization"] == "Bearer vertex-token"
            assert "x-goog-api-key" not in captured["headers"]
            backend.close()
