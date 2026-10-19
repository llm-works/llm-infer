# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""serve stops within a bounded time while a request is still running.

A stand-in `vllm serve` (``_shutdown_procs.py``) accepts a chat completion and
never answers, like a long generation. SIGTERM to serve must then:
- answer the waiting client with 503 at once,
- stop the vLLM process if serve started it (an external one keeps running),
- exit, with nothing left bound to the API port,
well inside a 15 s service stop timeout. With an external vLLM the upstream call
is still blocked at exit; the handler's daemon workers must not hold up exit.

Marked ``integration``: real processes, sockets and signals; no GPU or vLLM.
"""

from __future__ import annotations

import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration

PROCS = Path(__file__).with_name("_shutdown_procs.py")
REPO_ROOT = Path(__file__).resolve().parents[4]
STOP_BUDGET = 12.0  # Leaves margin under the smallest systemd TimeoutStopSec (15 s)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _port_bindable(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(predicate: Any, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.1)


def _healthy(api_port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{api_port}/health", timeout=1.0)
    except httpx.HTTPError:
        return False
    return r.status_code == 200 and r.json().get("status") == "ok"


def _reachable(url: str) -> bool:
    try:
        return httpx.get(url, timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


@dataclass
class Serve:
    proc: subprocess.Popen[bytes]
    api_port: int
    model_dir: Path
    auto_start: bool  # False: serve connects to a vLLM the test started


def _fake_vllm_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "vllm"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{PROCS}" vllm "$@"\n')
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


def _start_external_vllm(binary: Path, model_dir: Path, port: int) -> None:
    subprocess.Popen(
        [str(binary), "serve", str(model_dir), "--port", str(port)]
        + ["--served-model-name", model_dir.name],
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}/v1/models"
    _wait_for(lambda: _reachable(url), 30.0, "external vllm to listen")


@pytest.fixture(params=[True, False], ids=["auto-started", "external"])
def serve(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Serve]:
    auto_start: bool = request.param
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    binary = _fake_vllm_binary(tmp_path)
    api_port, vllm_port = _free_port(), _free_port()
    if not auto_start:
        _start_external_vllm(binary, model_dir, vllm_port)

    # Run as a script, so the repo root must be on the path to import llm_infer
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.Popen(
        [sys.executable, str(PROCS), "serve", str(model_dir), str(api_port)]
        + [str(vllm_port), str(binary), str(int(auto_start))],
        env=env,
        start_new_session=True,
    )
    try:
        _wait_for(lambda: _healthy(api_port), 60.0, "serve to become ready")
        yield Serve(proc, api_port, model_dir, auto_start)
    finally:
        _kill_leftovers(proc, model_dir)


def _kill_leftovers(proc: subprocess.Popen[bytes], model_dir: Path) -> None:
    """Never leave processes behind, whatever the test outcome."""
    for pgid_source in (proc.pid, _read_pid(model_dir)):
        if pgid_source is None:
            continue
        try:
            os.killpg(pgid_source, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.wait(timeout=5)


def _read_pid(model_dir: Path) -> int | None:
    pid_file = model_dir / "vllm.pid"
    return int(pid_file.read_text()) if pid_file.exists() else None


def _post_chat(api_port: int, stream: bool, result: dict[str, Any]) -> None:
    try:
        r = httpx.post(
            f"http://127.0.0.1:{api_port}/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": stream,
            },
            timeout=60.0,
        )
    except httpx.HTTPError as e:  # e.g. connection dropped by a killed server
        result["error"] = repr(e)
        return
    result["status"] = r.status_code
    result["text"] = r.text


@pytest.mark.parametrize(
    ("stream", "expected_status"),
    [
        (False, 503),
        # Headers (200) are sent before generation; the error arrives as an SSE event
        (True, 200),
    ],
)
def test_sigterm_with_request_in_flight_stops_within_budget(
    serve: Serve, stream: bool, expected_status: int
) -> None:
    vllm_pid = _read_pid(serve.model_dir)
    assert vllm_pid is not None
    result: dict[str, Any] = {}
    client = threading.Thread(target=_post_chat, args=(serve.api_port, stream, result))
    client.start()
    received = serve.model_dir / "request.received"
    _wait_for(received.exists, 30.0, "request to reach vllm")

    start = time.monotonic()
    serve.proc.send_signal(signal.SIGTERM)
    serve.proc.wait(timeout=STOP_BUDGET + 5)
    elapsed = time.monotonic() - start
    client.join(timeout=5.0)

    assert serve.proc.returncode == 0
    assert elapsed < STOP_BUDGET, f"serve took {elapsed:.1f}s to stop"
    assert result.get("status") == expected_status, result
    assert "Server shutting down" in result["text"]
    # serve stops the vLLM it started; an external one is not serve's to stop
    assert _pid_alive(vllm_pid) is not serve.auto_start
    assert _port_bindable(serve.api_port), "API port still bound after stop"
