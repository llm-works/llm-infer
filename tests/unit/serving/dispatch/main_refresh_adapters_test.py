# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for BootSequence._refresh_adapters: the rescan before a reacquire."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar
from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.config import InferenceConfig
from llm_infer.serving.dispatch.main import BootSequence

pytestmark = pytest.mark.unit

T = TypeVar("T")


class _RecordingLoopTasks:
    """LoopTasks stand-in: runs the call at once and records that it ran there."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def call(self, fn: Callable[[], T], timeout: float) -> T:
        self._calls.append("loop:start")
        result = fn()
        self._calls.append("loop:end")
        return result


def _boot() -> tuple[BootSequence, MagicMock, list[str]]:
    calls: list[str] = []
    boot = BootSequence(InferenceConfig(), MagicMock())
    boot._loop_tasks = _RecordingLoopTasks(calls)  # type: ignore[assignment]
    boot._engine = MagicMock()
    manager = MagicMock()
    manager.collect.side_effect = lambda: calls.append("collect") or ["adapter"]
    manager.apply.side_effect = lambda adapters: calls.append("apply") or len(adapters)
    boot._adapter_manager = manager
    return boot, manager, calls


def test_scan_off_loop_thread_swap_on_it_then_engine() -> None:
    boot, manager, calls = _boot()

    boot._refresh_adapters()

    # The slow collect runs before the loop call, only apply inside it
    assert calls == ["collect", "loop:start", "apply", "loop:end"]
    manager.apply.assert_called_once_with(["adapter"])
    boot._engine.set_adapters.assert_called_once_with(["adapter"])


def test_no_adapter_manager_is_a_noop() -> None:
    boot, _, calls = _boot()
    boot._adapter_manager = None

    boot._refresh_adapters()

    assert calls == []
    boot._engine.set_adapters.assert_not_called()
