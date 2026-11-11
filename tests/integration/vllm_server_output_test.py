# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Capture logging mode: reading a real child process's piped output."""

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

# Prints what `vllm serve` writes in capture mode: vLLM JSON records on stdout
# and another library's plain text on stderr, then exits
_CHILD = r"""
import json, sys
print(json.dumps({"levelname": "INFO", "name": "vllm.entrypoints",
                  "process": 7, "message": "starting"}), flush=True)
print("W1110 torch warning", file=sys.stderr, flush=True)
print(json.dumps({"levelname": "ERROR", "name": "vllm.engine",
                  "message": "boom", "exc_info": "Traceback ...\nError"}), flush=True)
"""


def _read_child(
    monkeypatch: pytest.MonkeyPatch, vllm_lg: MagicMock
) -> subprocess.Popen[bytes]:
    """Run the child, read its output with a reader logging to vllm_lg."""
    from llm_infer.engines import vllm_server_output
    from llm_infer.engines.vllm_server_output import VLLMOutputReader

    monkeypatch.setattr(vllm_server_output, "derive_lg", lambda lg, tags: vllm_lg)
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    reader = VLLMOutputReader(MagicMock(), proc.stdout)

    reader.start()
    proc.wait(timeout=30)
    reader.stop()
    return proc


def test_reader_logs_every_line_before_stop_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_lg = MagicMock()
    proc = _read_child(monkeypatch, vllm_lg)

    vllm_lg.info.assert_called_once_with(
        "starting", extra={"logger": "vllm.entrypoints", "pid": 7}
    )
    vllm_lg.debug.assert_called_once_with("W1110 torch warning", extra={})
    vllm_lg.error.assert_called_once_with(
        "boom", extra={"logger": "vllm.engine", "traceback": "Traceback ...\nError"}
    )
    assert proc.stdout.closed


def test_line_that_fails_to_log_does_not_stop_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead reader would leave vLLM blocked once the pipe is full."""
    vllm_lg = MagicMock()
    vllm_lg.info.side_effect = ValueError("rejected by the logger")

    _read_child(monkeypatch, vllm_lg)

    vllm_lg.warning.assert_called_once()
    assert vllm_lg.warning.call_args.args == ("vllm output line not logged",)
    vllm_lg.debug.assert_called_once()
    vllm_lg.error.assert_called_once()
