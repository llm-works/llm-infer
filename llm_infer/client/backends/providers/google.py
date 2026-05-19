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

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    MAX_BATCH_SIZE = 100

    def __init__(
        self,
        lg: Logger,
        api_key: str,
        model: str = "gemini-embedding-001",
        task_type: GoogleEmbeddingTaskType = GoogleEmbeddingTaskType.RETRIEVAL_DOCUMENT,
        ctx: BackendContext | None = None,
    ) -> None:
        """Initialize Google embedding backend.

        Args:
            lg: Logger instance.
            api_key: Google API key.
            model: Model name (default: gemini-embedding-001).
            task_type: Task type for optimized embeddings.
            ctx: Backend context with rate limiter and timeouts.
        """
        super().__init__(lg, model, ctx)
        self._api_key = api_key
        self._task_type = task_type

        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self._client = httpx.Client(timeout=self._ctx.request_timeout, headers=headers)
        self._async_client: httpx.AsyncClient | None = None
        self._headers = headers

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
            self._async_client = httpx.AsyncClient(
                timeout=self._ctx.request_timeout, headers=self._headers
            )
        return self._async_client

    # =========================================================================
    # Request building
    # =========================================================================

    def _build_single_request(
        self,
        text: str,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Build request payload for single embedding."""
        effective_model = model or self._model
        request: dict[str, Any] = {
            "model": f"models/{effective_model}",
            "content": {"parts": [{"text": text}]},
            "taskType": self._task_type.value,
        }
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
            "model": f"models/{effective_model}",
            "taskType": self._task_type.value,
        }
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
        url = f"{self.BASE_URL}/models/{effective_model}:embedContent"
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
        url = f"{self.BASE_URL}/models/{effective_model}:batchEmbedContents"
        payload = self._build_batch_request(texts, model, dimensions)
        return self._do_request_sync(url, payload)

    def _do_request_sync(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute sync HTTP request with error translation."""
        try:
            resp = self._client.post(url, json=payload)
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
        url = f"{self.BASE_URL}/models/{effective_model}:embedContent"
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
        url = f"{self.BASE_URL}/models/{effective_model}:batchEmbedContents"
        payload = self._build_batch_request(texts, model, dimensions)
        return await self._do_request_async(url, payload)

    async def _do_request_async(
        self, url: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute async HTTP request with error translation."""
        client = self._get_async_client()
        try:
            resp = await client.post(url, json=payload)
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
            prompt_tokens=None,
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
            total_prompt_tokens=None,
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
        return self._parse_single(data, model, dimensions)

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
                total_prompt_tokens=None,
            )
        if len(texts) > self.MAX_BATCH_SIZE:
            raise BackendRequestError(
                f"Batch size {len(texts)} exceeds maximum of {self.MAX_BATCH_SIZE}"
            )
        self._wait_rate_limit()
        data = self._execute_batch_sync(texts, model, dimensions)
        return self._parse_batch(data, len(texts), model, dimensions)

    async def embed_async(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        await self._wait_rate_limit_async()
        data = await self._execute_single_async(text, model, dimensions)
        return self._parse_single(data, model, dimensions)

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
                total_prompt_tokens=None,
            )
        if len(texts) > self.MAX_BATCH_SIZE:
            raise BackendRequestError(
                f"Batch size {len(texts)} exceeds maximum of {self.MAX_BATCH_SIZE}"
            )
        await self._wait_rate_limit_async()
        data = await self._execute_batch_async(texts, model, dimensions)
        return self._parse_batch(data, len(texts), model, dimensions)

    # =========================================================================
    # Token counting
    # =========================================================================

    def _count_tokens_sync(self, texts: list[str]) -> int:
        """Count tokens via API (sync)."""
        url = f"{self.BASE_URL}/models/{self._model}:countTokens"
        contents = [{"parts": [{"text": text}]} for text in texts]
        payload = {"contents": contents}
        data = self._do_request_sync(url, payload)
        total: int = data.get("totalTokens", 0)
        return total

    async def _count_tokens_async(self, texts: list[str]) -> int:
        """Count tokens via API (async)."""
        url = f"{self.BASE_URL}/models/{self._model}:countTokens"
        contents = [{"parts": [{"text": text}]} for text in texts]
        payload = {"contents": contents}
        data = await self._do_request_async(url, payload)
        total: int = data.get("totalTokens", 0)
        return total

    def count_tokens(self, text: str) -> int:
        self._wait_rate_limit()
        return self._count_tokens_sync([text])

    def count_tokens_batch(self, texts: list[str]) -> int:
        if not texts:
            return 0
        self._wait_rate_limit()
        return self._count_tokens_sync(texts)

    async def count_tokens_async(self, text: str) -> int:
        await self._wait_rate_limit_async()
        return await self._count_tokens_async([text])

    async def count_tokens_batch_async(self, texts: list[str]) -> int:
        if not texts:
            return 0
        await self._wait_rate_limit_async()
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
