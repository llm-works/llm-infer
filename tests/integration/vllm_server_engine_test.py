# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Stopping the vLLM server process, with real child processes."""

from unittest.mock import MagicMock

import pytest


class TestVLLMServerStop:
    """_stop_server escalates to SIGKILL after stop_timeout."""

    def _engine_with_process(self, proc: object, stop_timeout: float) -> object:
        from llm_infer.engines.vllm_server import VLLMServerEngine

        engine = object.__new__(VLLMServerEngine)
        engine._lg = MagicMock()
        engine._config = MagicMock(stop_timeout=stop_timeout)
        engine._process = proc
        return engine

    def test_sigkill_after_stop_timeout_when_sigterm_ignored(self) -> None:
        import subprocess
        import sys
        import time

        # Ignores SIGTERM like a server stuck in its shutdown; prints once ready
        script = (
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == b"ready"
        engine = self._engine_with_process(proc, stop_timeout=0.5)

        start = time.monotonic()
        engine._stop_server()
        elapsed = time.monotonic() - start

        assert proc.poll() is not None
        assert 0.5 <= elapsed < 5.0
        engine._lg.warning.assert_called_once_with(
            "vllm server did not stop gracefully, force killing"
        )

    def test_waits_for_whole_process_group(self) -> None:
        """A grandchild that outlives the main process (as vLLM's EngineCore can)
        is stopped too before _stop_server returns."""
        import os
        import subprocess
        import sys

        grandchild = (
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)"
        )
        # Main process: starts the grandchild, prints its pid, exits on SIGTERM
        script = (
            "import subprocess, sys, time; "
            f"p = subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
            "print(p.pid, flush=True); time.sleep(60)"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        assert proc.stdout is not None
        grandchild_pid = int(proc.stdout.readline())
        engine = self._engine_with_process(proc, stop_timeout=0.5)

        engine._stop_server()

        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
