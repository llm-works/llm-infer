"""Google Generative AI embedding backend.

Supports Google's gemini-embedding-001 and other embedding models via the
Generative AI API.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

import httpx
from appinfra.log import Logger

from ...errors import BackendRequestError, BackendTimeoutError, BackendUnavailableError
from ..auth import AuthProvider, GoogleAPIKeyHeaderAuth
from ..context import BackendContext
from ..embedding import Backend as EmbeddingBackend
from ..embedding import BatchEmbeddingResult, EmbeddingResult


class GoogleEmbeddingTaskType(StrEnum):
    """Google embedding task types for optimized embeddings."""

    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"


class GoogleEmbeddingBackend(EmbeddingBackend):
    """Google Generative AI embedding backend.

    Uses Google's embedContent and batchEmbedContents APIs.

    Example:
        backend = GoogleEmbeddingBackend(
            lg=logger,
            api_key="AIza...",
            model="gemini-embedding-001",
            task_type=GoogleEmbeddingTaskType.RETRIEVAL_DOCUMENT,
        )
        result = backend.embed("Hello world")
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    MAX_BATCH_SIZE = 100

    def __init__(
        self,
        lg: Logger,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        task_type: GoogleEmbeddingTaskType = GoogleEmbeddingTaskType.RETRIEVAL_DOCUMENT,
        ctx: BackendContext | None = None,
        count_tokens: bool = False,
        base_url: str | None = None,
        auth: AuthProvider | None = None,
    ) -> None:
        """Initialize Google embedding backend.

        Args:
            lg: Logger instance.
            api_key: Google API key (AI Studio). Wrapped as
                ``GoogleAPIKeyHeaderAuth`` if ``auth`` is not provided. Ignored
                when ``auth`` is provided.
            model: Model name (default: gemini-embedding-001).
            task_type: Task type for optimized embeddings.
            ctx: Backend context with rate limiter and timeouts.
            count_tokens: If True, populate prompt_tokens via countTokens API (extra call).
            base_url: API base URL. Defaults to the AI Studio endpoint; set to
                ``https://<region>-aiplatform.googleapis.com/v1/...`` for Vertex.
            auth: Auth provider. Takes precedence over ``api_key``. Use
                ``GCPServiceAccountAuth`` for Vertex.
        """
        super().__init__(lg, model, ctx)
        if auth is None and api_key is not None:
            auth = GoogleAPIKeyHeaderAuth(api_key)
        if auth is None:
            raise ValueError("Either api_key or auth must be provided")
        self._auth = auth
        self._task_type = task_type
        self._count_tokens = count_tokens
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

        self._client = httpx.Client(timeout=self._ctx.request_timeout)
        self._async_client: httpx.AsyncClient | None = None

    @property
    def provider(self) -> str:
        return "google"

    @property
    def task_type(self) -> GoogleEmbeddingTaskType:
        """Current task type for embeddings."""
        return self._task_type

    @property
    def max_batch_size(self) -> int:
        """Maximum batch size (100 for Google API)."""
        return self.MAX_BATCH_SIZE

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._ctx.request_timeout)
        return self._async_client

    def _build_headers(self) -> dict[str, str]:
        """Build request headers (sync)."""
        headers = {"Content-Type": "application/json"}
        headers.update(self._auth.headers())
        return headers

    async def _build_headers_async(self) -> dict[str, str]:
        """Build request headers (async, may refresh credentials off-loop)."""
        headers = {"Content-Type": "application/json"}
        headers.update(await self._auth.headers_async())
        return headers

    # =========================================================================
    # Request building
    # =========================================================================

    def _is_vertex(self) -> bool:
        """True when the backend is configured for Vertex AI.

        Vertex encodes the model name in the URL path and rejects requests
        that also carry a ``model`` field in the body (oneof conflict). AI
        Studio requires the body field. Detect once from ``base_url``.
        """
        return "aiplatform.googleapis.com" in self._base_url

    def _build_single_request(
        self,
        text: str,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Build request payload for single embedding."""
        effective_model = model or self._model
        request: dict[str, Any] = {
            "content": {"parts": [{"text": text}]},
            "taskType": self._task_type.value,
        }
        # AI Studio requires `model` in the body; Vertex rejects it (the model
        # is already in the URL path under publishers/google/models/<name>).
        if not self._is_vertex():
            request["model"] = f"models/{effective_model}"
        if dimensions is not None:
            request["outputDimensionality"] = dimensions
        return request

    def _build_batch_request(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Build request payload for batch embedding."""
        effective_model = model or self._model
        base_request: dict[str, Any] = {
            "taskType": self._task_type.value,
        }
        if not self._is_vertex():
            base_request["model"] = f"models/{effective_model}"
        if dimensions is not None:
            base_request["outputDimensionality"] = dimensions

        return {
            "requests": [
                {**base_request, "content": {"parts": [{"text": text}]}}
                for text in texts
            ]
        }

    # =========================================================================
    # HTTP execution
    # =========================================================================

    def _execute_single_sync(
        self,
        text: str,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Execute sync request for single embedding."""
        effective_model = model or self._model
        url = f"{self._base_url}/models/{effective_model}:embedContent"
        payload = self._build_single_request(text, model, dimensions)
        return self._do_request_sync(url, payload)

    def _execute_batch_sync(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Execute sync request for batch embedding."""
        effective_model = model or self._model
        url = f"{self._base_url}/models/{effective_model}:batchEmbedContents"
        payload = self._build_batch_request(texts, model, dimensions)
        return self._do_request_sync(url, payload)

    def _do_request_sync(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute sync HTTP request with error translation."""
        try:
            resp = self._client.post(url, json=payload, headers=self._build_headers())
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except httpx.ConnectError as e:
            raise BackendUnavailableError("Failed to connect to Google API") from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._ctx.request_timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Google API error: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    async def _execute_single_async(
        self,
        text: str,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Execute async request for single embedding."""
        effective_model = model or self._model
        url = f"{self._base_url}/models/{effective_model}:embedContent"
        payload = self._build_single_request(text, model, dimensions)
        return await self._do_request_async(url, payload)

    async def _execute_batch_async(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Execute async request for batch embedding."""
        effective_model = model or self._model
        url = f"{self._base_url}/models/{effective_model}:batchEmbedContents"
        payload = self._build_batch_request(texts, model, dimensions)
        return await self._do_request_async(url, payload)

    async def _do_request_async(
        self, url: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute async HTTP request with error translation."""
        client = self._get_async_client()
        try:
            headers = await self._build_headers_async()
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except httpx.ConnectError as e:
            raise BackendUnavailableError("Failed to connect to Google API") from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._ctx.request_timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Google API error: {e.response.text}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    # =========================================================================
    # Response parsing
    # =========================================================================

    def _parse_single(
        self,
        data: dict[str, Any],
        model: str | None = None,
        requested_dims: int | None = None,
        prompt_tokens: int | None = None,
    ) -> EmbeddingResult:
        """Parse response for single embedding."""
        try:
            embedding = data["embedding"]["values"]
            actual_dims = len(embedding)
        except (KeyError, TypeError) as e:
            raise BackendRequestError(f"Malformed response: {e}") from e
        if requested_dims is not None and actual_dims != requested_dims:
            raise BackendRequestError(
                f"Requested {requested_dims} dimensions but got {actual_dims}"
            )
        return EmbeddingResult(
            embedding=embedding,
            model=model or self._model,
            dimensions=actual_dims,
            prompt_tokens=prompt_tokens,
        )

    def _extract_embedding(
        self, emb: dict[str, Any], index: int, requested_dims: int | None
    ) -> tuple[list[float], int]:
        """Extract embedding vector and validate dimensions."""
        try:
            embedding: list[float] = emb["values"]
            dims = len(embedding)
        except (KeyError, TypeError) as e:
            raise BackendRequestError(
                f"Malformed response at index {index}: {e}"
            ) from e
        if requested_dims is not None and dims != requested_dims:
            raise BackendRequestError(
                f"Requested {requested_dims} dimensions but got {dims}"
            )
        return embedding, dims

    def _parse_batch(
        self,
        data: dict[str, Any],
        num_texts: int,
        model: str | None = None,
        requested_dims: int | None = None,
        total_prompt_tokens: int | None = None,
    ) -> BatchEmbeddingResult:
        """Parse batch embedding response."""
        raw_embeddings = data.get("embeddings", [])
        if len(raw_embeddings) != num_texts:
            raise BackendRequestError(
                f"Expected {num_texts} embeddings, got {len(raw_embeddings)}"
            )

        embeddings = []
        actual_dims = 0
        for i, emb in enumerate(raw_embeddings):
            embedding, dims = self._extract_embedding(emb, i, requested_dims)
            if i == 0:
                actual_dims = dims
            embeddings.append(embedding)

        return BatchEmbeddingResult(
            embeddings=embeddings,
            model=model or self._model,
            dimensions=actual_dims,
            size=num_texts,
            total_prompt_tokens=total_prompt_tokens,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        self._wait_rate_limit()
        data = self._execute_single_sync(text, model, dimensions)
        tokens = self._count_tokens_sync([text], model) if self._count_tokens else None
        return self._parse_single(data, model, dimensions, tokens)

    def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> BatchEmbeddingResult:
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=model or self._model,
                dimensions=0,
                size=0,
                total_prompt_tokens=0 if self._count_tokens else None,
            )
        if len(texts) > self.MAX_BATCH_SIZE:
            raise BackendRequestError(
                f"Batch size {len(texts)} exceeds maximum of {self.MAX_BATCH_SIZE}"
            )
        self._wait_rate_limit()
        data = self._execute_batch_sync(texts, model, dimensions)
        tokens = self._count_tokens_sync(texts, model) if self._count_tokens else None
        return self._parse_batch(data, len(texts), model, dimensions, tokens)

    async def embed_async(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        await self._wait_rate_limit_async()
        data = await self._execute_single_async(text, model, dimensions)
        tokens = (
            await self._count_tokens_async([text], model)
            if self._count_tokens
            else None
        )
        return self._parse_single(data, model, dimensions, tokens)

    async def embed_batch_async(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> BatchEmbeddingResult:
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=model or self._model,
                dimensions=0,
                size=0,
                total_prompt_tokens=0 if self._count_tokens else None,
            )
        if len(texts) > self.MAX_BATCH_SIZE:
            raise BackendRequestError(
                f"Batch size {len(texts)} exceeds maximum of {self.MAX_BATCH_SIZE}"
            )
        await self._wait_rate_limit_async()
        data = await self._execute_batch_async(texts, model, dimensions)
        tokens = (
            await self._count_tokens_async(texts, model) if self._count_tokens else None
        )
        return self._parse_batch(data, len(texts), model, dimensions, tokens)

    # =========================================================================
    # Token counting
    # =========================================================================

    def _count_tokens_sync(self, texts: list[str], model: str | None = None) -> int:
        """Count tokens via API (sync)."""
        effective_model = model or self._model
        url = f"{self._base_url}/models/{effective_model}:countTokens"
        contents = [{"parts": [{"text": text}]} for text in texts]
        payload = {"contents": contents}
        self._wait_rate_limit()
        data = self._do_request_sync(url, payload)
        total: int = data.get("totalTokens", 0)
        return total

    async def _count_tokens_async(
        self, texts: list[str], model: str | None = None
    ) -> int:
        """Count tokens via API (async)."""
        effective_model = model or self._model
        url = f"{self._base_url}/models/{effective_model}:countTokens"
        contents = [{"parts": [{"text": text}]} for text in texts]
        payload = {"contents": contents}
        await self._wait_rate_limit_async()
        data = await self._do_request_async(url, payload)
        total: int = data.get("totalTokens", 0)
        return total

    def count_tokens(self, text: str) -> int:
        return self._count_tokens_sync([text])

    def count_tokens_batch(self, texts: list[str]) -> int:
        if not texts:
            return 0
        return self._count_tokens_sync(texts)

    async def count_tokens_async(self, text: str) -> int:
        return await self._count_tokens_async([text])

    async def count_tokens_batch_async(self, texts: list[str]) -> int:
        if not texts:
            return 0
        return await self._count_tokens_async(texts)

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        self._client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
