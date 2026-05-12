"""Helper functions for LLMRouter routing logic.

Extracted from router.py to reduce duplication across the sync/async and
streaming/non-streaming chat methods.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any

from appinfra.dot_dict import DotDict

from .client import LLMClient
from .errors import BackendError
from .strategy import DecisionType, RoutingContext, RoutingDecision, RoutingResult
from .types import ChatRequest, ChatResponse

if TYPE_CHECKING:
    from .router import LLMRouter


def build_routing_context(
    request: ChatRequest,
    backend: str | None,
    role: str | None,
    context: RoutingContext | None,
) -> RoutingContext:
    """Build RoutingContext from request and routing parameters."""
    return RoutingContext(
        request=request,
        backend=backend or (context.backend if context else None),
        role=role or (context.role if context else None),
        metadata=context.metadata if context else DotDict(),
    )


def prepare_routing(
    router: LLMRouter,
    request: ChatRequest,
    backend: str | None,
    role: str | None,
    context: RoutingContext | None,
) -> tuple[ChatRequest, RoutingContext, RoutingDecision]:
    """Build routing context, get initial decision, and apply any request updates."""
    ctx = build_routing_context(request, backend, role, context)
    decision = get_initial_decision(router, ctx)
    request = decision.updated_request or request
    ctx.request = request
    return request, ctx, decision


def setup_routing(  # cq: max-lines=35
    router: LLMRouter,
    messages: list[dict[str, Any]],
    model: str | None,
    system: str | None,
    temperature: float,
    max_tokens: int | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    think: bool | None,
    adapter: str | None,
    context: dict[str, Any] | None,
    backend: str | None,
    role: str | None,
    routing_context: RoutingContext | None,
    **kwargs: Any,
) -> tuple[ChatRequest, RoutingContext, RoutingDecision]:
    """Build ChatRequest, routing context, and get initial decision.

    Combines ChatRequest creation with prepare_routing for concise method bodies.
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
        extra=kwargs or None,
        context=context,
    )
    return prepare_routing(router, request, backend, role, routing_context)


def _normalize_decision(
    router: LLMRouter,
    request: ChatRequest,
    decision: RoutingDecision,
) -> RoutingDecision:
    """Ensure decision has model resolved for target backend.

    Strategy may return backend-only decisions; this resolves the model
    so "auto"/"default" aliases work correctly for any backend.

    For fallback decisions, uses the target backend's default model rather
    than the original request's model (which may be provider-specific).
    """
    base_request = decision.updated_request or request

    # For fallback, use target backend's default model
    model_to_resolve = base_request.model
    if decision.decision_type == DecisionType.FALLBACK:
        model_to_resolve = None

    resolved = router.resolve(model=model_to_resolve, backend=decision.backend)
    updated = dataclasses.replace(base_request, model=resolved.model)
    return RoutingDecision(
        backend=resolved.backend,
        updated_request=updated,
        metadata=decision.metadata,
    )


def get_initial_decision(
    router: LLMRouter,
    context: RoutingContext,
) -> RoutingDecision:
    """Get initial routing decision from strategy or legacy resolution."""
    if router.strategy is not None:
        decision = router.strategy.select(router, context)
        if decision:
            if decision.backend in router.clients:
                return _normalize_decision(router, context.request, decision)
            router._lg.warning(
                "strategy returned invalid backend, falling back",
                extra={"backend": decision.backend, "available": list(router.clients)},
            )
    resolved = router.resolve(model=context.request.model, backend=context.backend)
    updated_request = dataclasses.replace(context.request, model=resolved.model)
    return RoutingDecision(backend=resolved.backend, updated_request=updated_request)


def make_result(
    backend_name: str,
    ctx: RoutingContext,
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
        context=ctx,
        decision=decision,
        response=response,
        error=error,
        metadata=DotDict(latency_ms=latency_ms),
    )


def handle_success(
    router: LLMRouter,
    response: ChatResponse,
    ctx: RoutingContext,
    decision: RoutingDecision,
    start_time: float,
) -> None:
    """Notify strategy of successful response."""
    if router.strategy:
        result = make_result(
            decision.backend, ctx, decision, start_time, response=response
        )
        router.strategy.on_result(router, result)


def _handle_error(
    router: LLMRouter,
    e: BackendError,
    ctx: RoutingContext,
    decision: RoutingDecision,
    start_time: float,
) -> RoutingDecision | None:
    """Handle error with strategy.

    Returns next decision if strategy wants to retry, None to raise the error.
    """
    lg = router._lg
    result = make_result(decision.backend, ctx, decision, start_time, error=e)
    if router.strategy:
        next_decision = router.strategy.on_error(router, result)
        if next_decision:
            if next_decision.backend not in router.clients:
                lg.warning(
                    "on_error returned invalid backend, not retrying",
                    extra={
                        "backend": next_decision.backend,
                        "available": list(router.clients),
                    },
                )
                return None
            same = next_decision.backend == decision.backend
            lg.warning(
                "backend failed, retrying" if same else "backend failed, trying next",
                extra={
                    "backend": decision.backend,
                    "error": str(e)[:200],
                    "next": next_decision.backend,
                },
            )
            return next_decision
    return None


class FallbackLoop:
    """Iterator for fallback retry loop.

    Handles timing, success notification, and retry decisions. Use in a for loop:

        for attempt in FallbackLoop(router, request, ctx, decision):
            try:
                response = attempt.client._chat(attempt.request)
                return attempt.success(response)
            except BackendError as e:
                attempt.fail(e)  # continues loop or re-raises
    """

    def __init__(
        self,
        router: LLMRouter,
        request: ChatRequest,
        ctx: RoutingContext,
        decision: RoutingDecision,
    ) -> None:
        self.router = router
        self.request = request
        self.ctx = ctx
        self.decision = decision
        self._done = False
        self._start_time = 0.0

    def __iter__(self) -> FallbackLoop:
        return self

    def __next__(self) -> FallbackLoop:
        if self._done:
            raise StopIteration
        self._start_time = time.monotonic()
        return self

    @property
    def client(self) -> LLMClient:
        return self.router._clients[self.decision.backend]

    def success(self, response: ChatResponse) -> ChatResponse:
        """Mark attempt as successful and notify strategy."""
        handle_success(self.router, response, self.ctx, self.decision, self._start_time)
        self._done = True
        return response

    def fail(self, e: BackendError) -> None:
        """Handle failure - either continues loop or re-raises."""
        next_decision = _handle_error(
            self.router, e, self.ctx, self.decision, self._start_time
        )
        if next_decision:
            self.decision = _normalize_decision(
                self.router, self.request, next_decision
            )
            self.request = self.decision.updated_request or self.request
            self.ctx.request = self.request
        else:
            raise e
