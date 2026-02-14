"""Multi-backend router for LLMClient.

LLMRouter routes requests to named LLMClient instances, enabling multi-backend
support with runtime backend selection.

Example:
    from appinfra.log import Logger
    from llm_infer.client import Factory

    lg = Logger("my-app")
    config = {
        "default": "local",
        "backends": {
            "local": {"type": "openai_compatible", "base_url": "http://localhost:8000/v1"},
            "openai": {"type": "openai", "base_url": "https://api.openai.com/v1"},
        },
    }
    router = Factory(lg).from_config(config)

    # Use default backend
    response = router.chat(messages)

    # Route to specific backend
    response = router.chat(messages, backend="openai")
"""

from __future__ import annotations

import types
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Self

from appinfra.log import Logger

from llm_infer.client.client import LLMClient
from llm_infer.client.types import ChatResponse


@dataclass(frozen=True)
class ResolvedTarget:
    """Resolved backend and model for a potential request.

    Returned by LLMRouter.resolve() to show which backend and model
    would be used for a request without actually making the call.

    Attributes:
        backend: Name of the backend that will handle the request.
        model: Model that will be used, or None if no default is configured.
    """

    backend: str
    model: str | None


class LLMRouter:
    """Multi-backend router - routes requests to named LLMClients.

    The router holds multiple LLMClient instances and routes requests based on
    the `backend` parameter. Each client can have its own configuration,
    throttling, and connection state.

    Attributes:
        clients: Dictionary mapping backend names to LLMClient instances.
        default: Name of the default backend.
    """

    def __init__(
        self,
        lg: Logger,
        clients: dict[str, LLMClient],
        default: str,
        model_to_backend: dict[str, str] | None = None,
    ) -> None:
        """Initialize the router with named clients.

        Args:
            lg: Logger instance.
            clients: Dictionary mapping backend names to LLMClient instances.
            default: Name of the default backend to use when not specified.
            model_to_backend: Optional mapping from model ID to backend name
                for intelligent model-based routing.

        Raises:
            ValueError: If default backend is not in clients dict.
        """
        self._lg = lg
        self._clients = clients
        self._default = default
        self._model_to_backend = model_to_backend or {}

        if default not in clients:
            raise ValueError(f"Default backend '{default}' not in clients")

        # Validate model routing references valid backends
        for model, backend_name in self._model_to_backend.items():
            if backend_name not in clients:
                raise ValueError(
                    f"Model '{model}' routes to unknown backend '{backend_name}'"
                )

    @property
    def clients(self) -> Mapping[str, LLMClient]:
        """Dictionary mapping backend names to LLMClient instances (read-only)."""
        return types.MappingProxyType(self._clients)

    @property
    def default(self) -> str:
        """Name of the default backend."""
        return self._default

    @property
    def models(self) -> Mapping[str, str]:
        """Mapping from model ID to backend name (read-only)."""
        return types.MappingProxyType(self._model_to_backend)

    def resolve(
        self, model: str | None = None, backend: str | None = None
    ) -> ResolvedTarget:
        """Resolve which backend and model will be used for a request.

        Performs the same routing logic as the chat methods but does not
        make an actual API call. Useful for:
        - Trace logging (show where request is going)
        - Dry-run mode (log what would be called)
        - Debugging routing decisions

        Resolution priority:
            1. Explicit backend name
            2. Model-based lookup (if model is in the routing table)
            3. Default backend

        Model resolution:
            - Uses explicit model if provided
            - Falls back to the target backend's default_model

        Args:
            model: Model to use (for routing and as the resolved model).
            backend: Explicit backend name (highest priority).

        Returns:
            ResolvedTarget with resolved backend name and model.

        Raises:
            ValueError: If explicit backend is not found.
        """
        # Resolve backend name
        if backend is not None:
            if backend not in self._clients:
                available = list(self._clients.keys())
                raise ValueError(
                    f"Backend '{backend}' not found. Available: {available}"
                )
            backend_name = backend
        elif model is not None and model in self._model_to_backend:
            backend_name = self._model_to_backend[model]
        else:
            backend_name = self._default

        # Resolve model (explicit or client's default)
        client = self._clients[backend_name]
        resolved_model = model if model is not None else client.default_model

        return ResolvedTarget(backend=backend_name, model=resolved_model)

    def get_client(
        self, backend: str | None = None, model: str | None = None
    ) -> LLMClient:
        """Get the client for the specified backend or model.

        Resolution priority:
            1. Explicit backend name
            2. Model-based lookup (if model is in the routing table)
            3. Default backend

        Args:
            backend: Backend name (highest priority).
            model: Model ID for model-based routing.

        Returns:
            The LLMClient for the resolved backend.

        Raises:
            ValueError: If explicit backend is not found.
        """
        resolved = self.resolve(model=model, backend=backend)
        return self._clients[resolved.backend]

    def can_call(self, backend: str | None = None, model: str | None = None) -> bool:
        """Check if a call is allowed for the specified backend (non-blocking).

        Delegates to the appropriate client's can_call() method based on
        backend/model routing.

        Args:
            backend: Backend name (highest priority).
            model: Model ID for model-based routing.

        Returns:
            True if a call is allowed, False if rate limited or in backoff.

        Raises:
            ValueError: If explicit backend is not found.
        """
        return self.get_client(backend, model).can_call()

    # =========================================================================
    # Sync API
    # =========================================================================

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return content (sync).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            backend: Backend to route to (uses default if not specified).
            **kwargs: Additional backend-specific parameters.

        Returns:
            Generated text content.
        """
        return self.get_client(backend, model).chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def chat_full(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request and return full response (sync).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            backend: Backend to route to (uses default if not specified).
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """
        return self.get_client(backend, model).chat_full(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream chat completion tokens (sync).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            backend: Backend to route to (uses default if not specified).
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        yield from self.get_client(backend, model).chat_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    # =========================================================================
    # Async API
    # =========================================================================

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return content (async).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            backend: Backend to route to (uses default if not specified).
            **kwargs: Additional backend-specific parameters.

        Returns:
            Generated text content.
        """
        return await self.get_client(backend, model).chat_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def chat_full_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request and return full response (async).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            backend: Backend to route to (uses default if not specified).
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """
        return await self.get_client(backend, model).chat_full_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def chat_stream_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system: str | None = None,
        adapter_id: str | None = None,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens (async).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            adapter_id: LoRA adapter name (OpenAI-compatible only).
            think: Enable thinking mode.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            backend: Backend to route to (uses default if not specified).
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        async for token in self.get_client(backend, model).chat_stream_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            adapter_id=adapter_id,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        ):
            yield token

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(self) -> None:
        """Close all clients (sync resources)."""
        for client in self._clients.values():
            try:
                client.close()
            except Exception as e:
                self._lg.warning("Error closing client", extra={"exception": e})

    async def aclose(self) -> None:
        """Close all clients (async resources)."""
        for client in self._clients.values():
            try:
                await client.aclose()
            except Exception as e:
                self._lg.warning("Error closing client", extra={"exception": e})

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
