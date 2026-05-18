"""Embedding client for generating vector embeddings via OpenAI-compatible API.

Provides sync and async APIs with retry support for transient errors.

Usage:
    from appinfra.log import Logger
    from llm_infer.client import EmbeddingClient
    from llm_infer.client.backends import RetryConfig

    lg = Logger("my-app")

    # Simple usage without retry
    with EmbeddingClient(lg, base_url="http://localhost:8001/v1") as client:
        result = client.embed("Hello world")
        print(result.embedding)  # list[float]

    # With retry for transient errors (retry for up to 2 minutes)
    retry = RetryConfig(timeout=120.0)
    with EmbeddingClient(lg, base_url="http://localhost:8001/v1", retry=retry) as client:
        results = client.embed_batch(["text1", "text2"])

    # Async usage
    async with EmbeddingClient(lg, base_url="http://localhost:8001/v1") as client:
        result = await client.embed_async("Hello world")
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Self

import httpx
from appinfra.log import Logger

from .backends import RetryConfig
from .errors import BackendRequestError, BackendTimeoutError, BackendUnavailableError
from .retry import RetryBase


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embedding: list[float]
    model: str
    prompt_tokens: int


class EmbeddingClient:
    """Client for generating embeddings via OpenAI-compatible API.

    Supports both sync and async operations with optional retry for transient
    errors (5xx, 429, 529).

    Attributes:
        model: The actual model name (discovered from server after first request).
    """

    def __init__(
        self,
        lg: Logger,
        base_url: str = "http://localhost:8001/v1",
        model: str = "default",
        timeout: float = 120.0,
        retry: RetryConfig | None = None,
    ) -> None:
        """Initialize the embeddings client.

        Args:
            lg: Logger for retry/error logging.
            base_url: Base URL for the embedding API (e.g., "http://localhost:8001/v1").
            model: Model name to send in requests (server may override).
            timeout: Request timeout in seconds (default matches chat clients).
            retry: Retry configuration for transient errors. None disables retry.
        """
        self._lg = lg
        self._base_url = base_url.rstrip("/")
        self._request_model = model
        self._discovered_model: str | None = None
        self._timeout = timeout
        self._retry = retry
        self._retry_base = RetryBase(lg)
        self._client = httpx.Client(timeout=timeout)
        self._async_client: httpx.AsyncClient | None = None

    @property
    def model(self) -> str:
        """Return the actual model name (discovered from server, or request model)."""
        return self._discovered_model or self._request_model

    def _update_model(self, response_model: str | None) -> None:
        """Cache the model name returned by the server."""
        if response_model and self._discovered_model is None:
            self._discovered_model = response_model

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout)
        return self._async_client

    # =========================================================================
    # HTTP execution with error translation
    # =========================================================================

    def _execute_sync(self, texts: str | list[str]) -> dict[str, Any]:
        """Execute sync request with error translation."""
        url = f"{self._base_url}/embeddings"
        payload = {"model": self._request_model, "input": texts}
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    async def _execute_async(self, texts: str | list[str]) -> dict[str, Any]:
        """Execute async request with error translation."""
        url = f"{self._base_url}/embeddings"
        payload = {"model": self._request_model, "input": texts}
        client = self._get_async_client()
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except httpx.ConnectError as e:
            raise BackendUnavailableError(
                f"Failed to connect to {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(
                f"Backend error: {e.response.text}", status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:
            raise BackendRequestError(f"Transport error: {e}") from e
        except json.JSONDecodeError as e:
            raise BackendRequestError(f"Invalid JSON response: {e}") from e

    # =========================================================================
    # Retry wrappers
    # =========================================================================

    def _call_with_retry(self, texts: str | list[str]) -> dict[str, Any]:
        """Execute request with retry on transient errors."""
        if self._retry is None:
            return self._execute_sync(texts)

        backoff = self._retry_base.create_backoff(self._retry)
        start_time = time.monotonic()
        retry_count = 0
        while True:
            try:
                return self._execute_sync(texts)
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._retry_base.should_retry(
                    e, start_time, self._retry.timeout
                ):
                    raise
                delay = self._retry_base.compute_delay(
                    backoff, self._retry.timeout, start_time
                )
                if delay is None:
                    raise
                retry_count += 1
                self._lg.warning(
                    "embedding request failed, retrying",
                    extra={"retry": retry_count, "delay": delay, "exception": e},
                )
                time.sleep(delay)

    async def _call_with_retry_async(self, texts: str | list[str]) -> dict[str, Any]:
        """Execute async request with retry on transient errors."""
        if self._retry is None:
            return await self._execute_async(texts)

        backoff = self._retry_base.create_backoff(self._retry)
        start_time = time.monotonic()
        retry_count = 0
        while True:
            try:
                return await self._execute_async(texts)
            except (BackendUnavailableError, BackendRequestError) as e:
                if not self._retry_base.should_retry(
                    e, start_time, self._retry.timeout
                ):
                    raise
                delay = self._retry_base.compute_delay(
                    backoff, self._retry.timeout, start_time
                )
                if delay is None:
                    raise
                retry_count += 1
                self._lg.warning(
                    "embedding request failed, retrying",
                    extra={"retry": retry_count, "delay": delay, "exception": e},
                )
                await asyncio.sleep(delay)

    # =========================================================================
    # Response parsing
    # =========================================================================

    def _parse_single_response(self, data: dict[str, Any]) -> EmbeddingResult:
        """Parse response for single embedding."""
        response_model = data.get("model", self._request_model)
        self._update_model(response_model)
        return EmbeddingResult(
            embedding=data["data"][0]["embedding"],
            model=response_model,
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
        )

    def _parse_batch_response(
        self, data: dict[str, Any], num_texts: int
    ) -> list[EmbeddingResult]:
        """Parse batch embedding response into EmbeddingResult list."""
        model = data.get("model", self.model)
        self._update_model(model)
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        tokens_per_text = prompt_tokens // num_texts if num_texts else 0

        results = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            results.append(
                EmbeddingResult(
                    embedding=item["embedding"],
                    model=model,
                    prompt_tokens=tokens_per_text,
                )
            )
        return results

    # =========================================================================
    # Model discovery
    # =========================================================================

    def discover(self) -> str:
        """Discover the actual model name from the server.

        Makes a lightweight probe request to get the model name.
        Caches the result for subsequent calls.

        Returns:
            The actual model name from the server.
        """
        if self._discovered_model is None:
            self.embed("")
        return self._discovered_model or self._request_model

    async def discover_async(self) -> str:
        """Discover the actual model name from the server (async).

        Makes a lightweight probe request to get the model name.
        Caches the result for subsequent calls.

        Returns:
            The actual model name from the server.
        """
        if self._discovered_model is None:
            await self.embed_async("")
        return self._discovered_model or self._request_model

    # =========================================================================
    # Sync API
    # =========================================================================

    def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            BackendUnavailableError: If the server is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the server returns an error.
        """
        data = self._call_with_retry(text)
        return self._parse_single_response(data)

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts in a single request.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult, one per input text.

        Raises:
            BackendUnavailableError: If the server is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the server returns an error.
        """
        if not texts:
            return []
        data = self._call_with_retry(texts)
        return self._parse_batch_response(data, len(texts))

    # =========================================================================
    # Async API
    # =========================================================================

    async def embed_async(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text (async version).

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            BackendUnavailableError: If the server is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the server returns an error.
        """
        data = await self._call_with_retry_async(text)
        return self._parse_single_response(data)

    async def embed_batch_async(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts in a single request (async).

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult, one per input text.

        Raises:
            BackendUnavailableError: If the server is unreachable.
            BackendTimeoutError: If the request times out.
            BackendRequestError: If the server returns an error.
        """
        if not texts:
            return []
        data = await self._call_with_retry_async(texts)
        return self._parse_batch_response(data, len(texts))

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        """Close the sync HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close all HTTP clients (sync and async)."""
        self._client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> Self:
        """Enter sync context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit sync context manager."""
        self.close()

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        await self.aclose()
