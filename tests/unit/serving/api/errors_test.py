# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for serving/api/errors.py status and channel-error mapping."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger
from appinfra.service.errors import (
    ChannelClosedError,
    ChannelError,
    ChannelTimeoutError,
)
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from llm_infer.serving.api.errors import raise_for_error_status, submit_or_timeout
from llm_infer.serving.dispatch.types import RequestStatus, Response

pytestmark = pytest.mark.unit


class _RaisingIPC:
    """IPC stub whose submit() raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def submit(self, request: Any) -> Any:
        raise self._exc


async def _submit(exc: Exception) -> JSONResponse:
    result = await submit_or_timeout(
        MagicMock(spec=Logger),
        _RaisingIPC(exc),
        Response(id="r1", status=RequestStatus.PENDING),
    )
    assert isinstance(result, JSONResponse)
    return result


def _error(resp: JSONResponse) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(bytes(resp.body))
    return body["error"]


@pytest.mark.asyncio
async def test_timeout_maps_to_504() -> None:
    resp = await _submit(ChannelTimeoutError("Request r1 timed out after 300s"))

    assert resp.status_code == 504
    assert _error(resp)["code"] == "timeout"


@pytest.mark.asyncio
async def test_closed_channel_maps_to_503() -> None:
    resp = await _submit(
        ChannelClosedError("Channel closed while waiting for response")
    )

    assert resp.status_code == 503
    assert _error(resp) == {
        "message": "Server shutting down",
        "type": "server_error",
        "code": "unavailable",
    }


@pytest.mark.asyncio
async def test_error_response_maps_to_500_with_message() -> None:
    """appinfra raises ChannelError when the response carries `error`."""
    resp = await _submit(ChannelError("Request failed: vLLM server error: boom"))

    assert resp.status_code == 500
    assert _error(resp)["message"] == "Request failed: vLLM server error: boom"


def test_unavailable_status_maps_to_503() -> None:
    with pytest.raises(HTTPException) as exc_info:
        raise_for_error_status(Response(id="r1", status=RequestStatus.UNAVAILABLE))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Server shutting down"
