# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for the admin API handlers and its config."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from appinfra.log import Logger
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from llm_infer.serving.admin.api import AdminApi, AdminStartupError, start
from llm_infer.serving.admin.controller import (
    ReleaseNotSupportedError,
    StateConflictError,
)
from llm_infer.serving.admin.state import BackendState
from llm_infer.serving.dispatch.config import AdminConfig, InferenceConfig

pytestmark = pytest.mark.unit

TOKEN = "secret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _client(controller: Any, tokens: list[str] | None = None) -> TestClient:
    cfg = AdminConfig(tokens=[TOKEN] if tokens is None else tokens)
    api = AdminApi(MagicMock(spec=Logger), controller, cfg)
    app = FastAPI()
    app.add_api_route("/admin/status", api.status, methods=["GET"])
    app.add_api_route("/admin/release", api.release, methods=["POST"])
    app.add_api_route("/admin/reacquire", api.reacquire, methods=["POST"])
    return TestClient(app)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": "Bearer "},
    ],
)
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/status"),
        ("POST", "/admin/release"),
        ("POST", "/admin/reacquire"),
    ],
)
def test_unauthorized(headers: dict[str, str], method: str, path: str) -> None:
    controller = MagicMock()

    r = _client(controller).request(method, path, headers=headers)

    assert r.status_code == 401
    controller.release.assert_not_called()
    controller.start_reacquire.assert_not_called()


def test_no_tokens_refuses_every_call() -> None:
    r = _client(MagicMock(), tokens=[]).get("/admin/status", headers=AUTH)

    assert r.status_code == 401


def test_any_configured_token_works() -> None:
    controller = MagicMock()
    controller.status.return_value = {"state": "serving", "last_error": None}
    client = _client(controller, tokens=["other", TOKEN])

    assert client.get("/admin/status", headers=AUTH).status_code == 200


def test_status() -> None:
    controller = MagicMock()
    controller.status.return_value = {"state": "released", "last_error": None}

    r = _client(controller).get("/admin/status", headers=AUTH)

    assert r.status_code == 200
    assert r.json() == {"state": "released", "last_error": None}


def test_release_returns_released() -> None:
    controller = MagicMock()
    controller.release.return_value = BackendState.RELEASED

    r = _client(controller).post("/admin/release", headers=AUTH)

    assert r.status_code == 200
    assert r.json() == {"state": "released"}


@pytest.mark.parametrize(
    "error",
    [StateConflictError("backend is loading"), ReleaseNotSupportedError("external")],
)
def test_release_conflict_is_409(error: Exception) -> None:
    controller = MagicMock()
    controller.release.side_effect = error

    r = _client(controller).post("/admin/release", headers=AUTH)

    assert r.status_code == 409
    assert r.json() == {"error": str(error)}


@pytest.mark.parametrize(
    ("state", "code"), [(BackendState.LOADING, 202), (BackendState.SERVING, 200)]
)
def test_reacquire(state: BackendState, code: int) -> None:
    controller = MagicMock()
    controller.start_reacquire.return_value = state

    r = _client(controller).post("/admin/reacquire", headers=AUTH)

    assert r.status_code == code
    assert r.json() == {"state": state.label}


# ---------------------------------------------------------------------------
# AdminConfig
# ---------------------------------------------------------------------------


def test_admin_config_defaults_disabled_on_localhost() -> None:
    cfg = InferenceConfig.from_dict({}).admin

    assert cfg.enabled is False
    assert cfg.host == "127.0.0.1"
    assert cfg.tokens == []


def test_admin_config_from_dict() -> None:
    raw = {"enabled": True, "port": 9001, "tokens": ["a", "b"], "retry_after": 5}

    cfg = InferenceConfig.from_dict({"admin": raw}).admin

    assert (cfg.enabled, cfg.port, cfg.tokens, cfg.retry_after) == (
        True,
        9001,
        ["a", "b"],
        5,
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"tokenz": ["a"]},  # unknown key
        {"tokens": [""]},  # empty token
        {"tokens": "a"},  # not a list
        {"port": 0},
        {"retry_after": -1},
    ],
)
def test_admin_config_rejects_bad_section(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.from_dict({"admin": raw})


# ---------------------------------------------------------------------------
# start() startup failure propagation
# ---------------------------------------------------------------------------


def test_start_raises_on_bind_failure() -> None:
    """If uvicorn fails to bind, start() raises AdminStartupError."""
    cfg = AdminConfig(port=9999, tokens=["t"])
    controller = MagicMock()
    lg = MagicMock(spec=Logger)

    mock_server = MagicMock()
    mock_server.start.side_effect = SystemExit("Address already in use")

    with patch("llm_infer.serving.admin.api._build_server", return_value=mock_server):
        with pytest.raises(AdminStartupError, match="failed to bind"):
            start(lg, controller, cfg)
