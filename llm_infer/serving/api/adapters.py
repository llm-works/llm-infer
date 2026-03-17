"""LoRA adapter API endpoints.

These endpoints communicate with the main process via IPC to manage adapters.
The main process holds the single source of truth for adapter state.
"""

from __future__ import annotations

import uuid
from typing import Any

from appinfra.log import Logger
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..dispatch.types import AdapterListRequest, AdapterRefreshRequest
from .errors import raise_for_error_status, submit_or_timeout


class AdapterInfo(BaseModel):
    """Information about a loaded adapter."""

    key: str = Field(..., description="Full adapter key (including version suffix)")
    name: str = Field(..., description="Logical adapter name without version suffix")
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


def _get_lg(request: Request) -> Logger:
    """Get logger from request state (injected by middleware)."""
    lg: Logger = request.state.lg
    return lg


async def _list_adapters(request: Request) -> AdapterListResponse | JSONResponse:
    """List all loaded adapters.

    Queries the main process for the current adapter list.
    """
    lg = _get_lg(request)
    ipc = _get_ipc(request)
    request_id = f"adapter-list-{uuid.uuid4().hex[:16]}"
    internal_request = AdapterListRequest(id=request_id)

    response = await submit_or_timeout(lg, ipc, internal_request)
    if isinstance(response, JSONResponse):
        return response
    raise_for_error_status(response)

    return AdapterListResponse(
        adapters=[
            AdapterInfo(
                key=a.key,
                name=a.name,
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
) -> RefreshResponse | JSONResponse:
    """Rescan adapter directory and reload enabled adapters.

    If key is provided, only refresh that specific adapter.
    Otherwise, rescan the entire directory.

    This operation is performed in the main process, ensuring the
    adapter state used for inference validation is updated.
    """
    lg = _get_lg(request)
    ipc = _get_ipc(request)
    request_id = f"adapter-refresh-{uuid.uuid4().hex[:16]}"
    internal_request = AdapterRefreshRequest(id=request_id, key=key)

    response = await submit_or_timeout(lg, ipc, internal_request)
    if isinstance(response, JSONResponse):
        return response
    raise_for_error_status(response)

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
        responses={504: {"description": "Gateway Timeout"}},
        summary="List loaded adapters",
        description="Returns all adapters that are enabled and loaded for inference.",
    )
    router.add_api_route(
        "/adapters/refresh",
        _refresh_adapters,
        methods=["POST"],
        response_model=RefreshResponse,
        responses={504: {"description": "Gateway Timeout"}},
        summary="Refresh adapters",
        description="Rescan adapter directory and reload enabled adapters.",
    )

    return router
