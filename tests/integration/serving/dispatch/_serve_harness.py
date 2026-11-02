# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Start llm-infer serve against the stand-in `vllm serve` from _serve_procs.py."""

from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

PROCS = Path(__file__).with_name("_serve_procs.py")
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_NAME = "fake-model"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def port_bindable(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for(predicate: Any, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.1)


def _ok(response: httpx.Response) -> bool:
    return response.status_code == 200


def _ready(response: httpx.Response) -> bool:
    return _ok(response) and response.json().get("status") == "ok"


def _reachable(url: str, check: Any = _ok) -> bool:
    try:
        return bool(check(httpx.get(url, timeout=1.0)))
    except httpx.HTTPError:
        return False


@dataclass
class Serve:
    proc: subprocess.Popen[bytes]
    api_port: int
    model_dir: Path
    auto_start: bool  # False: serve connects to a vLLM the test started

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def marker(self, name: str) -> Path:
        """Marker file written by the stand-in vLLM (see _serve_procs.py)."""
        return self.model_dir / name

    def vllm_pid(self) -> int | None:
        pid_file = self.model_dir / "vllm.pid"
        return int(pid_file.read_text()) if pid_file.exists() else None


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
    wait_for(lambda: _reachable(url), 30.0, "external vllm to listen")


@contextmanager
def running_serve(tmp_path: Path, **options: Any) -> Iterator[Serve]:
    """Run serve until the block ends; see _serve_procs.py for options."""
    auto_start = options.setdefault("auto_start", True)
    model_dir = tmp_path / MODEL_NAME
    model_dir.mkdir()
    binary = _fake_vllm_binary(tmp_path)
    api_port, vllm_port = free_port(), free_port()
    if not auto_start:
        _start_external_vllm(binary, model_dir, vllm_port)

    # Run as a script, so the repo root must be on the path to import llm_infer
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.Popen(
        [sys.executable, str(PROCS), "serve", str(model_dir), str(api_port)]
        + [str(vllm_port), str(binary), json.dumps(options)],
        env=env,
        start_new_session=True,
    )
    serve = Serve(proc, api_port, model_dir, auto_start)
    try:
        health = f"{serve.base_url}/health"
        wait_for(lambda: _reachable(health, _ready), 60.0, "serve to become ready")
        yield serve
    finally:
        _kill_leftovers(serve)


def _kill_leftovers(serve: Serve) -> None:
    """Never leave processes behind, whatever the test outcome."""
    for pgid_source in (serve.proc.pid, serve.vllm_pid()):
        if pgid_source is None:
            continue
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid_source, signal.SIGKILL)
    serve.proc.wait(timeout=5)


def chat_body(content: str, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
