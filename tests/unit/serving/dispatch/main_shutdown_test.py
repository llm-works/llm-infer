# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for BootSequence.shutdown ordering and bounds."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from llm_infer.serving.dispatch.config import InferenceConfig
from llm_infer.serving.dispatch.main import BootSequence

pytestmark = pytest.mark.unit


def _boot_with_mocks() -> tuple[BootSequence, MagicMock]:
    """BootSequence whose components report into one recorder, in call order."""
    config = InferenceConfig()
    config.dispatch.in_flight_join_timeout = 1.5
    config.dispatch.server_stop_timeout = 0.5
    boot = BootSequence(config, MagicMock())

    recorder = MagicMock()
    recorder.handler.join.return_value = True
    recorder.server.is_running = True
    boot._handler = recorder.handler
    boot._engine = recorder.engine
    boot._server = recorder.server
    boot._request_q = recorder.request_q
    boot._response_q = recorder.response_q
    return boot, recorder


def test_shutdown_order() -> None:
    """Handler answers first; the engine stops before in-flight calls are joined."""
    boot, recorder = _boot_with_mocks()

    boot.shutdown()

    # Drop truthiness checks (`if self._server and ...` records __bool__)
    calls = [c for c in recorder.mock_calls if not c[0].endswith("__bool__")]
    assert calls == [
        call.handler.shutdown(),
        call.engine.shutdown(),
        call.handler.join(1.5),
        call.server.stop(timeout=0.5),
        call.request_q.cancel_join_thread(),
        call.response_q.cancel_join_thread(),
    ]


def test_failing_step_does_not_skip_later_steps() -> None:
    boot, recorder = _boot_with_mocks()
    recorder.engine.shutdown.side_effect = RuntimeError("engine stuck")

    boot.shutdown()

    recorder.handler.join.assert_called_once()
    recorder.server.stop.assert_called_once()
    boot._lg.warning.assert_any_call(
        "shutdown error",
        extra={"step": "engine", "exception": recorder.engine.shutdown.side_effect},
    )


def test_unfinished_in_flight_calls_are_logged() -> None:
    boot, recorder = _boot_with_mocks()
    recorder.handler.join.return_value = False

    boot.shutdown()

    boot._lg.warning.assert_any_call(
        "in-flight requests still running at shutdown", extra={"timeout": 1.5}
    )
    recorder.server.stop.assert_called_once()
