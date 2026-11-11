# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Output of the `vllm serve` process llm-infer starts.

engines.vllm_server.logging.mode decides where it goes:

- stderr, stdout: the process writes to serve's stderr or stdout file
  descriptor directly, in vLLM's text format.
- vllm: vLLM's logging config file (VLLM_LOGGING_CONFIG_PATH) decides; output
  outside vLLM's loggers still goes to serve's stderr.
- drop: vLLM sets up no logging, its logger output is dropped; other output
  goes to serve's stderr.
- capture (experimental): vLLM logs JSON lines into a pipe serve reads, and
  serve logs each line again through its own logger. vLLM then depends on serve
  for its output: a reader that falls behind fills the pipe and blocks vLLM's
  next write, and with serve gone vLLM's writes fail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import IO, TYPE_CHECKING, Any

from appinfra.log import Logger, derive_lg

if TYPE_CHECKING:
    from ..serving.dispatch.config import VLLMServerLoggingConfig

# vLLM's JSON record level name -> Logger method; anything else logs at debug
_LEVEL_METHODS = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "critical",
}

# Child output after the process exits must not hold up a stop for long
_READER_JOIN_TIMEOUT = 5.0


def serve_env(cfg: VLLMServerLoggingConfig) -> dict[str, str]:
    """Environment for `vllm serve`: serve's own, with logging as configured.

    serve sets VLLM_CONFIGURE_LOGGING=0 for its own process; inherited, it
    leaves vLLM's loggers without handlers and their output is lost.
    """
    env = os.environ.copy()
    env["VLLM_CONFIGURE_LOGGING"] = "0" if cfg.mode == "drop" else "1"
    env.pop("VLLM_LOGGING_CONFIG_PATH", None)
    if cfg.mode == "vllm" and cfg.config_file is not None:
        env["VLLM_LOGGING_CONFIG_PATH"] = cfg.config_file
    return env


def serve_flags(cfg: VLLMServerLoggingConfig) -> list[str]:
    """`vllm serve` flags for the logging mode."""
    if cfg.mode == "capture":
        # One JSON object per record, traceback included: parsed, not guessed
        return ["--logging-config.formatter", "json"]
    return []


def serve_streams(cfg: VLLMServerLoggingConfig) -> tuple[Any, Any]:
    """Popen stdout and stderr for the logging mode."""
    if cfg.mode == "capture":
        # One pipe: lines of both streams stay in the order they were written
        return subprocess.PIPE, subprocess.STDOUT
    if cfg.mode == "stdout":
        return sys.stdout, sys.stdout
    return sys.stderr, sys.stderr


def parse_line(line: str, info_method: str) -> tuple[str, str, dict[str, Any]]:
    """Logger method, message and extra for one line of captured output.

    A vLLM JSON record keeps its level, except INFO, which is logged with
    info_method (logging.demote_info). Any other line (another library's
    output, a record split by an interleaved write) is logged as is at debug.
    """
    try:
        record = json.loads(line)
    except ValueError:
        record = None
    if not isinstance(record, dict) or not isinstance(record.get("message"), str):
        return "debug", line, {}

    levelname = str(record.get("levelname"))
    if levelname == "INFO":
        method = info_method
    else:
        method = _LEVEL_METHODS.get(levelname, "debug")
    extra: dict[str, Any] = {}
    # Renamed: Logger rejects extra keys that are LogRecord attributes
    # (name, process, exc_info)
    for key, name in (
        ("name", "logger"),
        ("vllm_process_name", "vllm_process"),
        ("process", "pid"),
        ("exc_info", "traceback"),
    ):
        if record.get(key) is not None:
            extra[name] = record[key]
    return method, record["message"], extra


class VLLMOutputReader:
    """Logs the piped output of a `vllm serve` process (capture mode).

    A thread reads lines until the process and its children close the pipe,
    which they do when they exit.
    """

    def __init__(self, lg: Logger, stream: IO[bytes], info_method: str) -> None:
        self._lg = derive_lg(lg, "vllm")
        self._stream = stream
        self._info_method = info_method
        self._thread = threading.Thread(
            target=self._run, name="vllm-output", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            for raw in self._stream:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._log(line)
        except (OSError, ValueError) as e:
            self._lg.warning("vllm output read failed", extra={"exception": e})

    def _log(self, line: str) -> None:
        """Log one line; a line that fails to log must not stop the reading,
        or vLLM blocks once the pipe is full."""
        method, message, extra = parse_line(line, self._info_method)
        try:
            getattr(self._lg, method)(message, extra=extra)
        except Exception as e:
            self._lg.warning(
                "vllm output line not logged", extra={"line": line, "exception": e}
            )

    def stop(self) -> None:
        """Wait for the remaining output, then close the pipe.

        Call once the process has exited. A child of it still holding the
        pipe open would keep the reader going; it is left to run as a daemon.
        """
        self._thread.join(_READER_JOIN_TIMEOUT)
        if self._thread.is_alive():
            self._lg.warning("vllm output reader still running after stop")
            return
        self._stream.close()
