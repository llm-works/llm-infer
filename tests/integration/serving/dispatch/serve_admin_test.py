# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""An admin releases the backend's GPU and reacquires it while serve keeps running.

serve runs against the stand-in `vllm serve` (``_serve_procs.py``) with the admin
listener enabled. Release must stop the vLLM process; inference answers 503 with
Retry-After meanwhile; reacquire starts a new vLLM process and inference answers
normally again, without restarting serve. With a vLLM serve didn't start,
release is refused.

Marked ``integration``: real processes and sockets; no GPU or vLLM.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from ._serve_harness import (
    Serve,
    chat_body,
    free_port,
    pid_alive,
    running_serve,
    wait_for,
)

pytestmark = pytest.mark.integration

TOKEN = "admin-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Admin:
    def __init__(self, port: int) -> None:
        self.url = f"http://127.0.0.1:{port}/admin"

    def post(self, action: str, headers: dict[str, str] = AUTH) -> httpx.Response:
        return httpx.post(f"{self.url}/{action}", headers=headers, timeout=30.0)

    def state(self) -> str:
        r = httpx.get(f"{self.url}/status", headers=AUTH, timeout=5.0)
        state: str = r.json()["state"]
        return state


@contextmanager
def _serve_with_admin(
    tmp_path: Path, auto_start: bool
) -> Iterator[tuple[Serve, Admin]]:
    port = free_port()
    admin_cfg = {"enabled": True, "port": port, "tokens": [TOKEN], "retry_after": 7}
    with running_serve(tmp_path, auto_start=auto_start, admin=admin_cfg) as serve:
        admin = Admin(port)
        wait_for(lambda: _reachable(admin), 10.0, "admin listener")
        yield serve, admin


def _reachable(admin: Admin) -> bool:
    try:
        return admin.state() == "serving"
    except httpx.HTTPError:
        return False


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[Serve, Admin]]:
    with _serve_with_admin(tmp_path, auto_start=True) as s:
        yield s


def _chat(serve: Serve) -> httpx.Response:
    return httpx.post(
        f"{serve.base_url}/v1/chat/completions",
        json=chat_body("quick", stream=False),
        timeout=30.0,
    )


def test_release_and_reacquire(served: tuple[Serve, Admin]) -> None:
    serve, admin = served
    first_vllm = serve.vllm_pid()
    assert first_vllm is not None and _chat(serve).status_code == 200

    r = admin.post("release")

    assert r.status_code == 200 and r.json() == {"state": "released"}
    assert not pid_alive(first_vllm), "vLLM still running after release"
    refused = _chat(serve)
    assert refused.status_code == 503
    assert refused.headers["retry-after"] == "7"
    assert admin.post("release").json() == {"state": "released"}  # idempotent

    start = time.monotonic()
    r = admin.post("reacquire")

    assert r.status_code == 202 and r.json() == {"state": "loading"}
    wait_for(lambda: admin.state() == "serving", 30.0, "backend serving again")
    second_vllm = serve.vllm_pid()
    assert second_vllm != first_vllm and pid_alive(second_vllm or 0)
    assert _chat(serve).status_code == 200
    assert serve.proc.poll() is None, "serve restarted"
    assert time.monotonic() - start < 30.0


def test_admin_calls_need_the_token(served: tuple[Serve, Admin]) -> None:
    _, admin = served

    r = admin.post("release", headers={"Authorization": "Bearer wrong"})

    assert r.status_code == 401
    assert admin.state() == "serving"


def test_release_refused_for_external_vllm(tmp_path: Path) -> None:
    with _serve_with_admin(tmp_path, auto_start=False) as (serve, admin):
        r = admin.post("release")

        assert r.status_code == 409
        assert admin.state() == "serving"
        assert _chat(serve).status_code == 200
