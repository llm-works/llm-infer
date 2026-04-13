"""Unit tests for serving/dispatch/errors.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger
from fastapi.responses import JSONResponse

from llm_infer.serving.dispatch.errors import ExceptionHandler

pytestmark = pytest.mark.unit


def _handler() -> ExceptionHandler:
    h = ExceptionHandler(lg=MagicMock(spec=Logger))
    return h


@pytest.mark.asyncio
async def test_handles_timeout_error() -> None:
    h = _handler()
    request = MagicMock()
    response = await h.handle(request, TimeoutError("ipc timeout"))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 504
    h._lg.error.assert_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_unknown_exception_reraised() -> None:
    h = _handler()
    request = MagicMock()
    with pytest.raises(ValueError, match="boom"):
        await h.handle(request, ValueError("boom"))


@pytest.mark.asyncio
async def test_trace_logged() -> None:
    h = _handler()
    request = MagicMock()
    await h.handle(request, TimeoutError("x"))
    # Trace was called with exception details
    h._lg.trace.assert_called()  # type: ignore[attr-defined]
