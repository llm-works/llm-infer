"""Helper functions for LLMRouter routing logic.

Extracted from router.py to reduce duplication across the sync/async and
streaming/non-streaming chat methods.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from appinfra.dot_dict import DotDict
from appinfra.log import Logger

from .errors import BackendError
from .strategy import RoutingContext, RoutingDecision, RoutingResult
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
    backend: str | None,
    role: str | None,
    context: RoutingContext | None,
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
    )
    return prepare_routing(router, request, backend, role, context)


def request_to_kwargs(request: ChatRequest, **extra: Any) -> dict[str, Any]:
    """Convert ChatRequest to kwargs dict for client calls."""
    return {
        "messages": request.messages,
        "model": request.model,
        "system": request.system,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "tools": request.tools,
        "tool_choice": request.tool_choice,
        "think": request.think,
        "adapter": request.adapter,
        **extra,
    }


def get_initial_decision(
    router: LLMRouter,
    context: RoutingContext,
) -> RoutingDecision:
    """Get initial routing decision from strategy or legacy resolution."""
    if router.strategy is not None:
        decision = router.strategy.select(router, context)
        if decision:
            if decision.backend in router.clients:
                return decision
            router._lg.warning(
                "strategy returned invalid backend, falling back",
                extra={"backend": decision.backend, "available": list(router.clients)},
            )
    resolved = router.resolve(model=context.request.model, backend=context.backend)
    return RoutingDecision(backend=resolved.backend)


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


def handle_error(
    router: LLMRouter,
    lg: Logger,
    e: BackendError,
    ctx: RoutingContext,
    decision: RoutingDecision,
    start_time: float,
) -> RoutingDecision | None:
    """Handle error with strategy.

    Returns next decision if strategy wants to retry, None to raise the error.
    """
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
