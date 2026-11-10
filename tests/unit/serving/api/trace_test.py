# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""serving/api/trace.py middleware helpers, called directly; requests through the
middleware are tested in integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger
from starlette.responses import StreamingResponse

from llm_infer.serving.api.trace import TraceMiddleware


def test_chunk_to_bytes_handles_all_types() -> None:
    """Direct test of _chunk_to_bytes for all branches."""
    lg = MagicMock(spec=Logger)
    mw = TraceMiddleware(app=lambda *a, **k: None)  # type: ignore[arg-type]
    assert mw._chunk_to_bytes(lg, b"abc") == b"abc"
    assert mw._chunk_to_bytes(lg, "abc") == b"abc"
    assert mw._chunk_to_bytes(lg, memoryview(b"abc")) == b"abc"
    assert mw._chunk_to_bytes(lg, bytearray(b"abc")) == b"abc"
    # Unknown type -> warning + str() encoded
    assert mw._chunk_to_bytes(lg, 42) == b"42"
    assert lg.warning.called


@pytest.mark.asyncio
async def test_log_response_reconstructs_body() -> None:
    """Direct test of _log_response: aggregates chunks and rebuilds Response."""
    lg = MagicMock(spec=Logger)
    mw = TraceMiddleware(app=lambda *a, **k: None)  # type: ignore[arg-type]

    async def gen():
        yield b"hello "
        yield "world"

    sr = StreamingResponse(gen(), status_code=200, media_type="text/plain")
    request = MagicMock()
    request.url.path = "/v1/test"

    result = await mw._log_response(lg, request, sr)
    assert result.status_code == 200
    assert result.body == b"hello world"
    assert result.media_type == "text/plain"
    # trace was called with assembled body
    trace_calls = [
        c for c in lg.trace.call_args_list if c.args and c.args[0] == "response"
    ]
    assert len(trace_calls) == 1
    assert trace_calls[0].kwargs["extra"]["body"] == "hello world"


@pytest.mark.asyncio
async def test_log_response_decode_error() -> None:
    """If response body cannot be decoded as utf-8, error is logged."""
    lg = MagicMock(spec=Logger)
    mw = TraceMiddleware(app=lambda *a, **k: None)  # type: ignore[arg-type]

    async def gen():
        yield b"\xff\xfe\xfd"

    sr = StreamingResponse(
        gen(), status_code=200, media_type="application/octet-stream"
    )
    request = MagicMock()
    request.url.path = "/v1/test"

    result = await mw._log_response(lg, request, sr)
    assert result.body == b"\xff\xfe\xfd"
    assert lg.error.called
