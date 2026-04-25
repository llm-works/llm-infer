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

import time
import types
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from appinfra.log import Logger

from . import router_helper as rh
from .base import BoundChatClient, ChatClient
from .client import LLMClient
from .errors import BackendError
from .resolver import ModelResolver
from .strategy import RoutingContext, RoutingStrategy
from .types import ChatResponse

if TYPE_CHECKING:
    from .discovery import ModelDiscovery

# Reserved model names that trigger special resolution logic
RESERVED_MODEL_NAMES = frozenset({"auto", "default"})


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
        model_to_backend: dict[str, str] | None = None,
        discovery: ModelDiscovery | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> None:
        """Initialize the router with named clients.

        Args:
            lg: Logger instance.
            clients: Dictionary mapping backend names to LLMClient instances.
            default: Name of the default backend to use when not specified.
            model_to_backend: Optional mapping from model ID to backend name
                for intelligent model-based routing. If discovery is provided,
                this is ignored (discovery manages the routing table).
            discovery: Optional ModelDiscovery for lazy model discovery.
                If provided, backends will be probed for models on first use.
            strategy: Optional routing strategy for fallback/ordering logic.

        Raises:
            ValueError: If default backend is not in clients dict.
        """
        self._lg = lg
        self._clients = clients
        self._default = default
        self._discovery = discovery
        self._strategy = strategy

        # Use discovery's routing table if available, otherwise use provided mapping
        if discovery is not None:
            model_map = dict(discovery.models)
        else:
            model_map = dict(model_to_backend) if model_to_backend else {}

        if default not in clients:
            raise ValueError(f"Default backend '{default}' not in clients")

        # Validate model routing references valid backends
        for model, backend_name in model_map.items():
            if backend_name not in clients:
                raise ValueError(
                    f"Model '{model}' routes to unknown backend '{backend_name}'"
                )

        self._resolver = ModelResolver(lg, model_map, default, discovery)

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
        return types.MappingProxyType(self._resolver.models)

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
        *,
        retry: bool = True,
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

        Reserved model names:
            - "auto": Probe the target backend, pick an available model
            - "default": Use the backend's configured default_model
              (if default_model="auto", falls back to auto logic)

        These reserved names skip routing lookup and are resolved on
        the target backend.

        Args:
            model: Model to use (for routing and as the resolved model).
            backend: Explicit backend name (highest priority).
            retry: If True (default) and client has backoff configured,
                retry on backend unavailable. Set to False to fail fast.

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
        elif model is not None and model not in RESERVED_MODEL_NAMES:
            backend_name = self._resolver.resolve_backend(model)
        else:
            backend_name = self._default

        # Resolve model on target backend
        client = self._clients[backend_name]
        resolved_model = self._resolver.resolve_model(client, model, retry=retry)

        return ResolvedTarget(backend=backend_name, model=resolved_model)

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
        )
        while True:
            start_time = time.monotonic()
            try:
                response = self._clients[decision.backend].chat(
                    **rh.request_to_kwargs(request, **kwargs)
                )
                rh.handle_success(self, response, ctx, decision, start_time)
                return response
            except BackendError as e:
                next_decision = rh.handle_error(
                    self, self._lg, e, ctx, decision, start_time
                )
                if next_decision:
                    decision = next_decision
                    request = decision.updated_request or request
                    continue
                raise

    def chat_stream(  # cq: max-lines=45
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
        )
        streamed = False
        while True:
            client = self._clients[decision.backend]
            start_time = time.monotonic()
            try:
                stream = client.chat_stream(**rh.request_to_kwargs(request, **kwargs))
                for token in stream:
                    streamed = True
                    yield token
                if client.last_response:
                    rh.handle_success(
                        self, client.last_response, ctx, decision, start_time
                    )
                return
            except BackendError as e:
                if streamed:
                    raise
                next_decision = rh.handle_error(
                    self, self._lg, e, ctx, decision, start_time
                )
                if next_decision:
                    decision = next_decision
                    request = decision.updated_request or request
                    continue
                raise

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
        )
        while True:
            start_time = time.monotonic()
            try:
                response = await self._clients[decision.backend].chat_async(
                    **rh.request_to_kwargs(request, **kwargs)
                )
                rh.handle_success(self, response, ctx, decision, start_time)
                return response
            except BackendError as e:
                next_decision = rh.handle_error(
                    self, self._lg, e, ctx, decision, start_time
                )
                if next_decision:
                    decision = next_decision
                    request = decision.updated_request or request
                    continue
                raise

    async def chat_stream_async(  # cq: max-lines=45
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
        )
        streamed = False
        while True:
            client = self._clients[decision.backend]
            start_time = time.monotonic()
            try:
                stream = client.chat_stream_async(
                    **rh.request_to_kwargs(request, **kwargs)
                )
                async for token in stream:
                    streamed = True
                    yield token
                if client.last_response:
                    rh.handle_success(
                        self, client.last_response, ctx, decision, start_time
                    )
                return
            except BackendError as e:
                if streamed:
                    raise
                next_decision = rh.handle_error(
                    self, self._lg, e, ctx, decision, start_time
                )
                if next_decision:
                    decision = next_decision
                    request = decision.updated_request or request
                    continue
                raise

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
