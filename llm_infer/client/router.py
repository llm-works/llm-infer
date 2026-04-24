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

from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from .base import ChatClient
from .client import LLMClient
from .errors import BackendError, BackendUnavailableError
from .strategy import RoutingContext, RoutingDecision, RoutingResult, RoutingStrategy
from .types import ChatRequest, ChatResponse

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
            self._model_to_backend = discovery.models
        else:
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
        """Mapping from model ID to backend name (read-only).

        Note: This returns the current known mappings. If lazy discovery is
        enabled, additional models may be discovered on first use.
        """
        if self._discovery is not None:
            return types.MappingProxyType(self._discovery.models)
        return types.MappingProxyType(self._model_to_backend)

    @property
    def discovery(self) -> ModelDiscovery | None:
        """Model discovery instance, if configured."""
        return self._discovery

    @property
    def strategy(self) -> RoutingStrategy | None:
        """Routing strategy, if configured."""
        return self._strategy

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
            # Try routing table for non-reserved model names
            backend_name = self._resolve_model_backend(model)
        else:
            # Reserved models or no model specified -> use default backend
            backend_name = self._default

        # Resolve model on target backend
        client = self._clients[backend_name]
        resolved_model = self._resolve_model(client, model, retry=retry)

        return ResolvedTarget(backend=backend_name, model=resolved_model)

    def _resolve_model_backend(self, model: str) -> str:
        """Resolve backend for a model, using lazy discovery if needed.

        Args:
            model: Model ID to look up.

        Returns:
            Backend name (falls back to default if not found).
        """
        # Check static routing table
        if model in self._model_to_backend:
            return self._model_to_backend[model]

        # Try lazy discovery if configured
        if self._discovery is not None:
            found = self._discovery.get_backend_for_model(model)
            if found is not None:
                # Refresh cached routing table. This is eventually consistent under
                # concurrent access (last write wins), but always returns correct
                # results since discovery.models is the source of truth.
                self._model_to_backend = self._discovery.models
                return found

        # Fall back to default
        return self._default

    def _resolve_model(
        self, client: LLMClient, model: str | None, *, retry: bool = True
    ) -> str | None:
        """Resolve model name, handling reserved names.

        Args:
            client: Target client to resolve model for.
            model: Model name (may be reserved like "auto" or "default").
            retry: If True and client has backoff, retry on backend failure.

        Returns:
            Resolved model name, or None if no model configured.
        """
        if model is None:
            return client.default_model

        if model == "default":
            default = client.default_model
            if default == "auto" or default is None:
                return self._resolve_auto_model(client, retry=retry)
            return default

        if model == "auto":
            return self._resolve_auto_model(client, retry=retry)

        return model

    def _list_models_with_retry(
        self, client: LLMClient, *, retry: bool = True
    ) -> list[str]:
        """List models from backend, retrying if backoff is configured.

        Args:
            client: Client to probe.
            retry: If True and client has backoff, retry on failure.

        Returns:
            List of model names.

        Raises:
            BackendUnavailableError: If backend unavailable after retries.
            Exception: For unexpected errors.
        """
        backoff = client.backoff
        timeout = client.timeout
        start_time = time.time()

        while True:
            try:
                models = client.backend.list_models()
                if backoff is not None:
                    backoff.reset()
                return models
            except BackendUnavailableError as e:
                if backoff is None or not retry:
                    raise

                # Check timeout
                elapsed = time.time() - start_time
                if timeout > 0 and elapsed >= timeout:
                    self._lg.error(
                        "model discovery timed out",
                        extra={"error": str(e), "elapsed": elapsed},
                    )
                    raise

                delay = backoff.next_delay()
                self._lg.warning(
                    "backend unavailable for model discovery, retrying",
                    extra={"error": str(e), "delay": delay, "elapsed": elapsed},
                )
                time.sleep(delay)

    def _resolve_auto_model(
        self, client: LLMClient, *, retry: bool = True
    ) -> str | None:
        """Resolve "auto" to an actual model by probing the backend.

        Resolution order:
            1. If only one model available, use it
            2. If backend has configured default_model (non-auto), use it
            3. Use first model from list_models()

        If retry=True and client has backoff configured, retries on
        BackendUnavailableError until success or timeout.

        Args:
            client: Client to probe for available models.
            retry: If True and client has backoff, retry on failure.

        Returns:
            Resolved model name, or None if no models available.
        """
        try:
            models = self._list_models_with_retry(client, retry=retry)
        except BackendUnavailableError as e:
            self._lg.warning(
                "failed to discover models for auto resolution",
                extra={"error": str(e)},
            )
            return client.default_model if client.default_model != "auto" else None
        except Exception as e:
            self._lg.warning(
                "failed to discover models for auto resolution",
                extra={"exception": e},
            )
            return client.default_model if client.default_model != "auto" else None

        if not models:
            return None

        if len(models) == 1:
            return models[0]

        # Multiple models: prefer configured default if valid
        default = client.default_model
        if default and default != "auto" and default in models:
            return default

        return models[0]

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
    # Strategy helpers
    # =========================================================================

    def _get_initial_decision(self, context: RoutingContext) -> RoutingDecision:
        """Get initial routing decision.

        If strategy is set, asks strategy for first backend.
        Otherwise uses legacy resolution.
        """
        if self._strategy is not None:
            decision = self._strategy.select(self, context)
            if decision and decision.backend in self._clients:
                return decision

        # Legacy resolution
        resolved = self.resolve(model=context.request.model, backend=context.backend)
        return RoutingDecision(backend=resolved.backend)

    def _make_result(
        self,
        backend_name: str,
        routing_context: RoutingContext,
        decision: RoutingDecision,
        start_time: float,
        response: ChatResponse | None = None,
        error: BackendError | None = None,
    ) -> RoutingResult:
        """Create a RoutingResult for strategy callbacks."""
        latency_ms = (time.monotonic() - start_time) * 1000
        return RoutingResult(
            backend=backend_name,
            model=response.model if response else None,
            context=routing_context,
            decision=decision,
            response=response,
            error=error,
            metadata=DotDict(latency_ms=latency_ms),
        )

    # =========================================================================
    # Sync API
    # =========================================================================

    def chat(  # cq: max-lines=70
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

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            think: Enable thinking mode.
            adapter: LoRA adapter name (OpenAI-compatible only).
            backend: Backend to route to (uses default if not specified).
            role: Application-defined role for strategy routing (e.g., "summarize").
            context: Routing context for strategy-based routing.
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """
        request = ChatRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
        )
        ctx = RoutingContext(
            request=request,
            backend=backend or (context.backend if context else None),
            role=role or (context.role if context else None),
            metadata=context.metadata if context else DotDict(),
        )
        decision = self._get_initial_decision(ctx)
        request = decision.updated_request or request

        while True:
            client = self._clients[decision.backend]
            start_time = time.monotonic()

            try:
                response = client.chat(
                    messages=request.messages,
                    model=request.model,
                    system=request.system,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    think=request.think,
                    adapter=request.adapter,
                    **kwargs,
                )
                result = self._make_result(
                    decision.backend,
                    ctx,
                    decision,
                    start_time,
                    response=response,
                )
                if self._strategy:
                    self._strategy.on_result(self, result)
                return response

            except BackendError as e:
                result = self._make_result(
                    decision.backend, ctx, decision, start_time, error=e
                )
                if self._strategy:
                    next_decision = self._strategy.on_error(self, result)
                    if next_decision:
                        self._lg.warning(
                            "backend failed, trying next",
                            extra={
                                "backend": decision.backend,
                                "error": str(e)[:200],
                                "next": next_decision.backend,
                            },
                        )
                        decision = next_decision
                        request = decision.updated_request or request
                        continue
                raise

    def chat_stream(  # cq: max-lines=75
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

        Note: Fallback only occurs before streaming starts. Once streaming
        begins, errors are raised immediately (no mid-stream fallback).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            think: Enable thinking mode.
            adapter: LoRA adapter name (OpenAI-compatible only).
            backend: Backend to route to (uses default if not specified).
            role: Application-defined role for strategy routing (e.g., "summarize").
            context: Routing context for strategy-based routing.
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        request = ChatRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
        )
        ctx = RoutingContext(
            request=request,
            backend=backend or (context.backend if context else None),
            role=role or (context.role if context else None),
            metadata=context.metadata if context else DotDict(),
        )
        decision = self._get_initial_decision(ctx)
        request = decision.updated_request or request

        while True:
            client = self._clients[decision.backend]
            start_time = time.monotonic()

            try:
                # Start the stream - fallback only happens here, before first token
                stream = client.chat_stream(
                    messages=request.messages,
                    model=request.model,
                    system=request.system,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    think=request.think,
                    adapter=request.adapter,
                    **kwargs,
                )
                # Once streaming starts, yield tokens without fallback
                yield from stream

                # Stream completed successfully
                result = self._make_result(
                    decision.backend,
                    ctx,
                    decision,
                    start_time,
                    response=client.last_response,
                )
                if self._strategy:
                    self._strategy.on_result(self, result)
                return

            except BackendError as e:
                result = self._make_result(
                    decision.backend, ctx, decision, start_time, error=e
                )
                if self._strategy:
                    next_decision = self._strategy.on_error(self, result)
                    if next_decision:
                        self._lg.warning(
                            "backend failed, trying next",
                            extra={
                                "backend": decision.backend,
                                "error": str(e)[:200],
                                "next": next_decision.backend,
                            },
                        )
                        decision = next_decision
                        request = decision.updated_request or request
                        continue
                raise

    # =========================================================================
    # Async API
    # =========================================================================

    async def chat_async(  # cq: max-lines=70
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

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            think: Enable thinking mode.
            adapter: LoRA adapter name (OpenAI-compatible only).
            backend: Backend to route to (uses default if not specified).
            role: Application-defined role for strategy routing (e.g., "summarize").
            context: Routing context for strategy-based routing.
            **kwargs: Additional backend-specific parameters.

        Returns:
            ChatResponse with content, usage, thinking, tool_calls, etc.
        """
        request = ChatRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
        )
        ctx = RoutingContext(
            request=request,
            backend=backend or (context.backend if context else None),
            role=role or (context.role if context else None),
            metadata=context.metadata if context else DotDict(),
        )
        decision = self._get_initial_decision(ctx)
        request = decision.updated_request or request

        while True:
            client = self._clients[decision.backend]
            start_time = time.monotonic()

            try:
                response = await client.chat_async(
                    messages=request.messages,
                    model=request.model,
                    system=request.system,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    think=request.think,
                    adapter=request.adapter,
                    **kwargs,
                )
                result = self._make_result(
                    decision.backend,
                    ctx,
                    decision,
                    start_time,
                    response=response,
                )
                if self._strategy:
                    self._strategy.on_result(self, result)
                return response

            except BackendError as e:
                result = self._make_result(
                    decision.backend, ctx, decision, start_time, error=e
                )
                if self._strategy:
                    next_decision = self._strategy.on_error(self, result)
                    if next_decision:
                        self._lg.warning(
                            "backend failed, trying next",
                            extra={
                                "backend": decision.backend,
                                "error": str(e)[:200],
                                "next": next_decision.backend,
                            },
                        )
                        decision = next_decision
                        request = decision.updated_request or request
                        continue
                raise

    async def chat_stream_async(  # cq: max-lines=75
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

        Note: Fallback only occurs before streaming starts. Once streaming
        begins, errors are raised immediately (no mid-stream fallback).

        Args:
            messages: List of chat messages.
            model: Model to use (overrides default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Control tool use.
            think: Enable thinking mode.
            adapter: LoRA adapter name (OpenAI-compatible only).
            backend: Backend to route to (uses default if not specified).
            role: Application-defined role for strategy routing (e.g., "summarize").
            context: Routing context for strategy-based routing.
            **kwargs: Additional backend-specific parameters.

        Yields:
            String tokens as they arrive.
        """
        request = ChatRequest(
            messages=messages,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            think=think,
            adapter=adapter,
        )
        ctx = RoutingContext(
            request=request,
            backend=backend or (context.backend if context else None),
            role=role or (context.role if context else None),
            metadata=context.metadata if context else DotDict(),
        )
        decision = self._get_initial_decision(ctx)
        request = decision.updated_request or request

        while True:
            client = self._clients[decision.backend]
            start_time = time.monotonic()

            try:
                # Start the stream - fallback only happens here, before first token
                stream = client.chat_stream_async(
                    messages=request.messages,
                    model=request.model,
                    system=request.system,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    think=request.think,
                    adapter=request.adapter,
                    **kwargs,
                )
                # Once streaming starts, yield tokens without fallback
                async for token in stream:
                    yield token

                # Stream completed successfully
                result = self._make_result(
                    decision.backend,
                    ctx,
                    decision,
                    start_time,
                    response=client.last_response,
                )
                if self._strategy:
                    self._strategy.on_result(self, result)
                return

            except BackendError as e:
                result = self._make_result(
                    decision.backend, ctx, decision, start_time, error=e
                )
                if self._strategy:
                    next_decision = self._strategy.on_error(self, result)
                    if next_decision:
                        self._lg.warning(
                            "backend failed, trying next",
                            extra={
                                "backend": decision.backend,
                                "error": str(e)[:200],
                                "next": next_decision.backend,
                            },
                        )
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
