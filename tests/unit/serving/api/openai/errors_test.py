"""Unit tests for serving/api/openai/errors.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from llm_infer.serving.api.openai.errors import (
    ErrorDetail,
    ErrorResponse,
    OpenAIHTTPException,
    create_error_response,
    http_status_to_error_type,
    openai_exception_handler,
)

pytestmark = pytest.mark.unit


class TestHttpStatusToErrorType:
    @pytest.mark.parametrize(
        "code,expected",
        [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (429, "rate_limit_error"),
            (500, "server_error"),
            (503, "service_unavailable"),
        ],
    )
    def test_known_codes(self, code: int, expected: str) -> None:
        assert http_status_to_error_type(code) == expected

    def test_unknown_code_defaults_to_server_error(self) -> None:
        assert http_status_to_error_type(418) == "server_error"


class TestCreateErrorResponse:
    def test_basic(self) -> None:
        resp = create_error_response(404, "not found")
        assert resp.status_code == 404
        body = json.loads(resp.body.decode())
        assert body["error"]["message"] == "not found"
        assert body["error"]["type"] == "not_found_error"

    def test_with_param_and_code(self) -> None:
        resp = create_error_response(
            400, "bad input", param="model", code="invalid_value"
        )
        body = json.loads(resp.body.decode())
        assert body["error"]["param"] == "model"
        assert body["error"]["code"] == "invalid_value"


@pytest.mark.asyncio
async def test_openai_exception_handler_v1_path() -> None:
    request = MagicMock()
    request.url.path = "/v1/chat/completions"
    exc = HTTPException(status_code=400, detail="bad request")
    resp = await openai_exception_handler(request, exc)
    assert resp.status_code == 400
    body = json.loads(resp.body.decode())
    assert body["error"]["message"] == "bad request"
    assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_openai_exception_handler_non_v1_path() -> None:
    """Non-/v1 paths use default FastAPI behavior."""
    request = MagicMock()
    request.url.path = "/health"
    exc = HTTPException(status_code=404, detail="not found")
    resp = await openai_exception_handler(request, exc)
    body = json.loads(resp.body.decode())
    assert body == {"detail": "not found"}


def test_openai_http_exception_with_extras() -> None:
    exc = OpenAIHTTPException(
        status_code=400, message="bad", param="model", code="invalid"
    )
    assert exc.status_code == 400
    assert exc.detail == "bad"
    assert exc.param == "model"
    assert exc.code == "invalid"


def test_error_models_serialization() -> None:
    detail = ErrorDetail(message="x", type="server_error")
    response = ErrorResponse(error=detail)
    data = response.model_dump()
    assert data["error"]["message"] == "x"
