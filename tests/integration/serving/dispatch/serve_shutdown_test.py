# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""serve stops within a bounded time while a request is still running.

A stand-in `vllm serve` (``_serve_procs.py``) accepts a chat completion and
never answers, like a long generation. SIGTERM to serve must then:
- answer the waiting client with 503 at once,
- stop the vLLM process if serve started it (an external one keeps running),
- exit, with nothing left bound to the API port,
well inside a 15 s service stop timeout. With an external vLLM the upstream call
is still blocked at exit; the handler's daemon workers must not hold up exit.

Marked ``integration``: real processes, sockets and signals; no GPU or vLLM.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ._serve_harness import (
    Serve,
    chat_body,
    pid_alive,
    port_bindable,
    running_serve,
    wait_for,
)

pytestmark = pytest.mark.integration

STOP_BUDGET = 12.0  # Leaves margin under the smallest systemd TimeoutStopSec (15 s)


@pytest.fixture(params=[True, False], ids=["auto-started", "external"])
def serve(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Serve]:
    with running_serve(tmp_path, auto_start=request.param) as s:
        yield s


def _post_chat(base_url: str, stream: bool, result: dict[str, Any]) -> None:
    try:
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            json=chat_body("hi", stream),
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
    vllm_pid = serve.vllm_pid()
    assert vllm_pid is not None
    result: dict[str, Any] = {}
    client = threading.Thread(target=_post_chat, args=(serve.base_url, stream, result))
    client.start()
    wait_for(serve.marker("request.received").exists, 30.0, "request to reach vllm")

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
    assert pid_alive(vllm_pid) is not serve.auto_start
    assert port_bindable(serve.api_port), "API port still bound after stop"
