"""LoRA adapter API endpoints."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..adapters import AdapterManager


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


def _get_manager(request: Request) -> AdapterManager:
    """Get adapter manager from app state."""
    manager = getattr(request.app.state, "adapter_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Adapter management not enabled. Set lora.enabled=true and lora.base_path in config.",
        )
    return cast(AdapterManager, manager)


async def _list_adapters(request: Request) -> AdapterListResponse:
    """List all loaded adapters."""
    manager = _get_manager(request)
    adapters = manager.list()
    return AdapterListResponse(
        adapters=[AdapterInfo(**manager.to_dict(a)) for a in adapters],
        count=len(adapters),
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
    """
    manager = _get_manager(request)

    if adapter_id:
        # Refresh single adapter
        adapter = manager.refresh_one(adapter_id)
        return RefreshResponse(
            adapter_id=adapter_id,
            adapters_loaded=len(manager.list()),
            status="loaded" if adapter else "unloaded",
        )
    else:
        # Full rescan
        count = manager.scan()
        return RefreshResponse(
            adapter_id=None,
            adapters_loaded=count,
            status="scanned",
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
