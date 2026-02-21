"""Route handlers for the inference API.

Uses appinfra's FastAPI framework with IPC channel for subprocess communication.
"""

import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, Request

from ..dispatch.metrics import format_metrics_for_api
from ..dispatch.types import MetricsRequest, MetricsResponse
from ..dispatch.types import Request as InternalRequest
from .errors import raise_for_error_status
from .schemas import GenerateRequest, GenerateResponse, HealthResponse


def create_health_handler(
    ready_flag: Any = None,
) -> Callable[[], Coroutine[Any, Any, HealthResponse]]:
    """Create health check handler with optional ready flag.

    Args:
        ready_flag: Optional multiprocessing.Value for readiness state.
                   If None, always returns "ok".
    """

    async def health_handler() -> HealthResponse:
        """Health check endpoint."""
        if ready_flag is not None and not ready_flag.value:
            return HealthResponse(status="initializing")
        return HealthResponse(status="ok")

    return health_handler


async def health_handler() -> HealthResponse:
    """Health check endpoint (legacy, always returns ok)."""
    return HealthResponse(status="ok")


async def _handle_generate(body: GenerateRequest, ipc: Any) -> GenerateResponse:
    """Handle generate request submission and response."""
    request_id = str(uuid.uuid4())
    internal_request = InternalRequest(
        id=request_id,
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        use_chat_template=body.use_chat_template,
    )

    response = await ipc.submit(request_id, internal_request)
    raise_for_error_status(response)

    return GenerateResponse(
        text=response.result or "",
        prompt_tokens=response.prompt_tokens or 0,
        completion_tokens=response.completion_tokens or 0,
    )


def create_routes(model_name: str) -> APIRouter:
    """
    Create the main API router with inference endpoints.

    Args:
        model_name: Name of the loaded model for metadata.

    Returns:
        APIRouter with /generate endpoint.
    """
    router = APIRouter()

    @router.post("/generate", response_model=GenerateResponse)
    async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        """Generate text from a prompt."""
        return await _handle_generate(body, request.app.state.ipc_channel)

    @router.get("/metrics")
    async def metrics(request: Request, reset_peak: bool = False) -> dict:
        """Get server metrics including GPU memory and KV cache usage."""
        ipc = request.app.state.ipc_channel
        request_id = str(uuid.uuid4())
        metrics_request = MetricsRequest(id=request_id, reset_peak=reset_peak)
        response: MetricsResponse = await ipc.submit(request_id, metrics_request)
        return format_metrics_for_api(response)

    return router
