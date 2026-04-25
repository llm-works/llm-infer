"""Multi-backend router for LLMClient.

LLMRouter routes requests to named LLMClient instances, enabling multi-backend
support with runtime backend selection.

Model discovery is lazy - backends are only probed when first used. This avoids
startup errors when backends are configured but not running.

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

from . import router_helper as rh
from .base import ChatClient
from .bound import BoundChatClient
from .client import LLMClient
from .discovery import RESERVED_MODEL_NAMES, ModelDiscovery
from .errors import BackendError
from .strategy import RoutingContext, RoutingStrategy
from .types import ChatResponse


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


class LLMRouter(ChatClient):
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
        discovery: ModelDiscovery | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> None:
        """Initialize the router with named clients.

        Args:
            lg: Logger instance.
            clients: Dictionary mapping backend names to LLMClient instances.
            default: Name of the default backend to use when not specified.
            discovery: Optional ModelDiscovery for model→backend routing.
                If provided, enables model-based routing to backends.
            strategy: Optional routing strategy for fallback/ordering logic.

        Raises:
            ValueError: If default backend is not in clients dict.
        """
        self._lg = lg
        self._clients = clients
        self._default = default
        self._discovery = discovery
        self._strategy = strategy

        if default not in clients:
            raise ValueError(f"Default backend '{default}' not in clients")

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
        """Mapping from model ID to backend name (read-only).

        Note: This returns the current known mappings. If lazy discovery is
        enabled, additional models may be discovered on first use.
        """
        if self._discovery is not None:
            return types.MappingProxyType(self._discovery.models)
        return types.MappingProxyType({})

    @property
    def discovery(self) -> ModelDiscovery | None:
        """Model discovery instance, if configured."""
        return self._discovery

    @property
    def strategy(self) -> RoutingStrategy | None:
        """Routing strategy, if configured."""
        return self._strategy

    def with_chat_args(self, **kwargs: Any) -> BoundChatClient:
        """Create a bound ChatClient with kwargs merged into every call.

        Returns a BoundChatClient that wraps this router and merges the
        provided kwargs into every chat call. Useful for binding routing
        parameters (role, backend) without passing them each time.

        Args:
            **kwargs: Arguments to merge into every chat call (e.g., role,
                backend, model, temperature).

        Returns:
            BoundChatClient wrapping this router with bound kwargs.

        Example:
            router = Factory(lg).from_config(config)
            exploration = router.with_chat_args(role="exploration")
            synthesis = router.with_chat_args(role="synthesis")

            # Both implement ChatClient, can be used interchangeably
            exploration.chat(messages)  # role="exploration" merged
            synthesis.chat(messages)    # role="synthesis" merged
        """
        return BoundChatClient(self, **kwargs)

    def resolve(
        self,
        model: str | None = None,
        backend: str | None = None,
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

        Reserved model names ('auto', 'default') are resolved by the target
        client's ModelResolver. This method will probe the backend if needed.

        Args:
            model: Model to use (for routing and as the resolved model).
            backend: Explicit backend name (highest priority).

        Returns:
            ResolvedTarget with resolved backend name and model.

        Raises:
            ValueError: If explicit backend is not found.
        """
        backend_name = self._resolve_backend(model, backend)
        client = self._clients[backend_name]

        # Use client's resolver to resolve "auto"/"default"
        resolved_model = client._resolve_model(model)

        return ResolvedTarget(backend=backend_name, model=resolved_model)

    def _resolve_backend(self, model: str | None, backend: str | None) -> str:
        """Resolve which backend to use for a request.

        Args:
            model: Model name (used for model-based routing).
            backend: Explicit backend name (highest priority).

        Returns:
            Backend name to use.

        Raises:
            ValueError: If explicit backend is not found.
        """
        # Explicit backend takes priority
        if backend is not None:
            if backend not in self._clients:
                available = list(self._clients.keys())
                raise ValueError(
                    f"Backend '{backend}' not found. Available: {available}"
                )
            return backend

        # Model-based routing (skip reserved names)
        if model is not None and model not in RESERVED_MODEL_NAMES:
            if self._discovery is not None:
                found = self._discovery.get_backend_for_model(model)
                if found is not None:
                    return found

        # Fall back to default
        return self._default

    def get_client(
        self, model: str | None = None, backend: str | None = None
    ) -> LLMClient:
        """Get the client for the specified backend or model.

        Resolution priority:
            1. Explicit backend name
            2. Model-based lookup (if model is in the routing table)
            3. Default backend

        Args:
            model: Model ID for model-based routing.
            backend: Backend name (highest priority).

        Returns:
            The LLMClient for the resolved backend.

        Raises:
            ValueError: If explicit backend is not found.
        """
        resolved = self.resolve(model=model, backend=backend)
        return self._clients[resolved.backend]

    def can_call(self, model: str | None = None, backend: str | None = None) -> bool:
        """Check if a call is allowed for the specified backend (non-blocking).

        Delegates to the appropriate client's can_call() method based on
        backend/model routing.

        Args:
            model: Model ID for model-based routing.
            backend: Backend name (highest priority).

        Returns:
            True if a call is allowed, False if rate limited or in backoff.

        Raises:
            ValueError: If explicit backend is not found.
        """
        return self.get_client(model=model, backend=backend).can_call()

    # =========================================================================
    # Sync API
    # =========================================================================

    def chat(  # cq: max-lines=35
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        backend: str | None = None,
        role: str | None = None,
        context: RoutingContext | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request (sync).

        See ChatClient.chat() for common parameters. Router-specific args:
            backend: Backend to route to (uses default if not specified).
            role: Application-defined role for strategy routing.
            context: Routing context for strategy-based routing.
        """
        request, ctx, decision = rh.setup_routing(
            self,
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            backend,
            role,
            context,
            **kwargs,
        )
        for attempt in rh.FallbackLoop(self, request, ctx, decision):
            try:
                return attempt.success(attempt.client._chat(attempt.request))
            except BackendError as e:
                attempt.fail(e)
        raise RuntimeError("FallbackLoop exhausted without result")

    def chat_stream(  # cq: max-lines=35
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        backend: str | None = None,
        role: str | None = None,
        context: RoutingContext | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream chat completion tokens (sync).

        Fallback only occurs before streaming starts. Once streaming begins,
        errors are raised immediately. See ChatClient.chat_stream() for common
        parameters. Router-specific: backend, role, context.
        """
        request, ctx, decision = rh.setup_routing(
            self,
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            backend,
            role,
            context,
            **kwargs,
        )
        streamed = False
        for attempt in rh.FallbackLoop(self, request, ctx, decision):
            try:
                for token in attempt.client._chat_stream(attempt.request):
                    streamed = True
                    yield token
                if attempt.client.last_response:
                    attempt.success(attempt.client.last_response)
                return
            except BackendError as e:
                if streamed:
                    raise
                attempt.fail(e)

    # =========================================================================
    # Async API
    # =========================================================================

    async def chat_async(  # cq: max-lines=35
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        backend: str | None = None,
        role: str | None = None,
        context: RoutingContext | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request (async).

        See ChatClient.chat_async() for common parameters. Router-specific args:
            backend: Backend to route to (uses default if not specified).
            role: Application-defined role for strategy routing.
            context: Routing context for strategy-based routing.
        """
        request, ctx, decision = rh.setup_routing(
            self,
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            backend,
            role,
            context,
            **kwargs,
        )
        for attempt in rh.FallbackLoop(self, request, ctx, decision):
            try:
                return attempt.success(
                    await attempt.client._chat_async(attempt.request)
                )
            except BackendError as e:
                attempt.fail(e)
        raise RuntimeError("FallbackLoop exhausted without result")

    async def chat_stream_async(  # cq: max-lines=35
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        think: bool | None = None,
        adapter: str | None = None,
        backend: str | None = None,
        role: str | None = None,
        context: RoutingContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens (async).

        Fallback only occurs before streaming starts. Once streaming begins,
        errors are raised immediately. See ChatClient.chat_stream_async() for
        common parameters. Router-specific: backend, role, context.
        """
        request, ctx, decision = rh.setup_routing(
            self,
            messages,
            model,
            system,
            temperature,
            max_tokens,
            tools,
            tool_choice,
            think,
            adapter,
            backend,
            role,
            context,
            **kwargs,
        )
        streamed = False
        for attempt in rh.FallbackLoop(self, request, ctx, decision):
            try:
                async for token in attempt.client._chat_stream_async(attempt.request):
                    streamed = True
                    yield token
                if attempt.client.last_response:
                    attempt.success(attempt.client.last_response)
                return
            except BackendError as e:
                if streamed:
                    raise
                attempt.fail(e)

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
