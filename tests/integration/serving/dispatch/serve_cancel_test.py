# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""A request the API side gives up on stops consuming the backend.

The stand-in `vllm serve` (``_serve_procs.py``) holds a request open until its
client closes the connection, as vLLM does for a long generation, and records
the close (vLLM aborts the generation on disconnect). With max_concurrent=1, a
follow-up request only gets through if the abandoned one freed its slot.

Covers both ways the API side gives up: response_timeout (504, or an SSE
timeout event when streaming), and the client disconnecting.

Marked ``integration``: real processes and sockets; no GPU or vLLM.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from ._serve_harness import Serve, chat_body, running_serve, wait_for

pytestmark = pytest.mark.integration

RESPONSE_TIMEOUT = 2.0
CANCEL_WITHIN = 5.0  # From the API giving up to the backend seeing the close


@pytest.fixture
def serve(tmp_path: Path) -> Iterator[Serve]:
    with running_serve(
        tmp_path, response_timeout=RESPONSE_TIMEOUT, max_concurrent=1
    ) as s:
        yield s


def _assert_upstream_closed_and_slot_free(serve: Serve) -> None:
    wait_for(serve.marker("request.closed").exists, CANCEL_WITHIN, "upstream close")
    r = httpx.post(
        f"{serve.base_url}/v1/chat/completions",
        json=chat_body("quick", stream=False),
        timeout=10.0,
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize("stream", [False, True])
def test_timeout_cancels_upstream_request(serve: Serve, stream: bool) -> None:
    r = httpx.post(
        f"{serve.base_url}/v1/chat/completions",
        json=chat_body("hold", stream),
        timeout=30.0,
    )

    if stream:
        assert r.status_code == 200  # headers precede generation
        assert '"code": "timeout"' in r.text or '"code":"timeout"' in r.text
    else:
        assert r.status_code == 504
    _assert_upstream_closed_and_slot_free(serve)


@pytest.mark.parametrize("stream", [False, True])
def test_client_disconnect_cancels_upstream_request(serve: Serve, stream: bool) -> None:
    # The client gives up (read timeout 1 s) before serve's RESPONSE_TIMEOUT,
    # so only the disconnect can trigger the cancel
    with (
        pytest.raises(httpx.ReadTimeout),
        httpx.Client(timeout=httpx.Timeout(10.0, read=1.0)) as client,
    ):
        r = client.post(
            f"{serve.base_url}/v1/chat/completions",
            json=chat_body("hold", stream),
        )
        for _ in r.iter_bytes():  # streaming: headers arrive, then nothing
            pass

    assert serve.marker("request.received").exists()
    _assert_upstream_closed_and_slot_free(serve)
