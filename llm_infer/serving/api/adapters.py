"""LoRA adapter API endpoints.

These endpoints communicate with the main process via IPC to manage adapters.
The main process holds the single source of truth for adapter state.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..dispatch.types import AdapterListRequest, AdapterRefreshRequest


class AdapterInfo(BaseModel):
    """Information about a loaded adapter."""

    key: str = Field(..., description="Adapter lookup key")
    description: str | None = Field(None, description="Optional description")
    loaded_at: str = Field(..., description="ISO timestamp when adapter was loaded")
    md5: str | None = Field(
        None, description="MD5 hash of weights file (first 12 chars)"
    )
    mtime: str | None = Field(
        None, description="ISO timestamp of weights file modification"
    )


class AdapterListResponse(BaseModel):
    """Response containing list of loaded adapters."""

    adapters: list[AdapterInfo] = Field(default_factory=list)
    count: int = Field(..., description="Number of loaded adapters")


class RefreshResponse(BaseModel):
    """Response from refresh operation."""

    key: str | None = Field(None, description="Adapter key if single refresh")
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

    try:
        response = await ipc.submit(request_id, internal_request)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list adapters: {e}"
        ) from e

    return AdapterListResponse(
        adapters=[
            AdapterInfo(
                key=a.key,
                description=a.description,
                loaded_at=a.loaded_at,
                md5=a.md5,
                mtime=a.mtime,
            )
            for a in response.adapters
        ],
        count=len(response.adapters),
    )


async def _refresh_adapters(
    request: Request,
    key: str | None = Query(
        default=None, description="Specific adapter key to refresh"
    ),
) -> RefreshResponse:
    """Rescan adapter directory and reload enabled adapters.

    If key is provided, only refresh that specific adapter.
    Otherwise, rescan the entire directory.

    This operation is performed in the main process, ensuring the
    adapter state used for inference validation is updated.
    """
    ipc = _get_ipc(request)
    request_id = f"adapter-refresh-{uuid.uuid4().hex[:16]}"
    internal_request = AdapterRefreshRequest(id=request_id, key=key)

    try:
        response = await ipc.submit(request_id, internal_request)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to refresh adapters: {e}"
        ) from e

    return RefreshResponse(
        key=response.key,
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
