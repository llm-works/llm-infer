#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

# ci-run:
# ci-requires: port:8111
# ci-timeout: 30

"""Spawn the retry mock server, then run the retry client, then clean up.

Exercises the paired demo end-to-end so CI catches wiring regressions.
Run the individual scripts manually for the pedagogical two-terminal flow.
"""

import socket
import subprocess
import sys
import time

_SERVER_MOD = "llm_infer.examples.client.test_retry_server"
_CLIENT_MOD = "llm_infer.examples.client.test_retry_client"
_HOST = "localhost"
_PORT = 8111


def _wait_for_bind(host: str, port: int, timeout_s: float) -> bool:
    """Poll host:port until a TCP connect succeeds or the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main() -> int:
    """Coordinate server + client subprocesses; return client's exit code."""
    server = subprocess.Popen(
        [sys.executable, "-m", _SERVER_MOD],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_bind(_HOST, _PORT, timeout_s=10):
            print(f"server did not bind {_HOST}:{_PORT}", file=sys.stderr)
            return 1
        return subprocess.run(
            [sys.executable, "-m", _CLIENT_MOD], check=False
        ).returncode
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
