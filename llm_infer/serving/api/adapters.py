"""LoRA adapter API endpoints.

These endpoints communicate with the main process via IPC to manage adapters.
The main process holds the single source of truth for adapter state.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from ..dispatch.types import AdapterListRequest, AdapterRefreshRequest


class AdapterInfo(BaseModel):
    """Information about a loaded adapter."""

    adapter_id: str = Field(..., description="Unique identifier")
    description: str | None = Field(None, description="Optional description")
    loaded_at: str = Field(..., description="ISO timestamp when adapter was loaded")


class AdapterListResponse(BaseModel):
    """Response containing list of loaded adapters."""

    adapters: list[AdapterInfo] = Field(default_factory=list)
    count: int = Field(..., description="Number of loaded adapters")


class RefreshResponse(BaseModel):
    """Response from refresh operation."""

    adapter_id: str | None = Field(None, description="Adapter ID if single refresh")
    adapters_loaded: int = Field(
        ..., description="Number of enabled adapters after refresh"
    )
    status: str = Field(..., description="Result: 'loaded', 'unloaded', or 'scanned'")


def _get_ipc(request: Request) -> Any:
    """Get IPC channel from app state."""
    return request.app.state.ipc_channel


async def _list_adapters(request: Request) -> AdapterListResponse:
    """List all loaded adapters.

    Queries the main process for the current adapter list.
    """
    ipc = _get_ipc(request)
    request_id = f"adapter-list-{uuid.uuid4().hex[:16]}"
    internal_request = AdapterListRequest(id=request_id)

    response = await ipc.submit(request_id, internal_request)

    return AdapterListResponse(
        adapters=[
            AdapterInfo(
                adapter_id=a.adapter_id,
                description=a.description,
                loaded_at=a.loaded_at,
            )
            for a in response.adapters
        ],
        count=len(response.adapters),
    )


async def _refresh_adapters(
    request: Request,
    adapter_id: str | None = Query(
        default=None, description="Specific adapter to refresh"
    ),
) -> RefreshResponse:
    """Rescan adapter directory and reload enabled adapters.

    If adapter_id is provided, only refresh that specific adapter.
    Otherwise, rescan the entire directory.

    This operation is performed in the main process, ensuring the
    adapter state used for inference validation is updated.
    """
    ipc = _get_ipc(request)
    request_id = f"adapter-refresh-{uuid.uuid4().hex[:16]}"
    internal_request = AdapterRefreshRequest(id=request_id, adapter_id=adapter_id)

    response = await ipc.submit(request_id, internal_request)

    return RefreshResponse(
        adapter_id=response.adapter_id,
        adapters_loaded=response.adapters_loaded,
        status=response.status,
    )


def create_adapter_router() -> APIRouter:
    """Create the adapter management router."""
    router = APIRouter(tags=["Adapters"])

    router.add_api_route(
        "/adapters",
        _list_adapters,
        methods=["GET"],
        response_model=AdapterListResponse,
        summary="List loaded adapters",
        description="Returns all adapters that are enabled and loaded for inference.",
    )
    router.add_api_route(
        "/adapters/refresh",
        _refresh_adapters,
        methods=["POST"],
        response_model=RefreshResponse,
        summary="Refresh adapters",
        description="Rescan adapter directory and reload enabled adapters.",
    )

    return router
