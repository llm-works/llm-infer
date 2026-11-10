# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Concurrent warmup: steps run on warmup's worker threads at the same time."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.dispatch.warmup import (
    WarmupResult,
    warmup_adapters,
    warmup_base_model,
)


@pytest.fixture
def lg() -> Logger:
    return MagicMock(spec=Logger)


def _adapter(key: str) -> MagicMock:
    a = MagicMock()
    a.key = key
    return a


class _ConcurrentEngine:
    """Thread-safe stand-in engine that records how many generate calls overlap.

    Each call waits on a barrier of `parties`, so a test only passes when that
    many calls really are in flight together.
    """

    max_model_len = 4096

    def __init__(self, parties: int, fail_at: int | None = None) -> None:
        self._barrier = threading.Barrier(parties)
        self._lock = threading.Lock()
        self._active = 0
        self._fail_at = fail_at
        self.peak = 0
        self.calls: list[int] = []

    def supports_embeddings(self) -> bool:
        return False

    def generate(self, prompt: str, max_tokens: int, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.calls.append(max_tokens)
        self._barrier.wait(timeout=5)
        with self._lock:
            self._active -= 1
        if max_tokens == self._fail_at:
            raise RuntimeError("boom")
        return {"finish_reason": "stop"}


class TestConcurrentWarmup:
    def test_base_sweep_runs_all_steps_at_once(self, lg: Logger) -> None:
        e = _ConcurrentEngine(parties=4)
        results = warmup_base_model(lg, e, 4)
        assert [r.max_tokens for r in results] == [32, 128, 512, 2048]
        assert e.peak == 4

    def test_base_sweep_capped_at_concurrency(self, lg: Logger) -> None:
        e = _ConcurrentEngine(parties=2)
        results = warmup_base_model(lg, e, 2)
        assert len(results) == 4
        assert e.peak == 2

    def test_base_failure_cuts_baseline_at_failed_step(self, lg: Logger) -> None:
        """Same baseline as the sequential sweep: steps before the first failure."""
        e = _ConcurrentEngine(parties=4, fail_at=128)
        results = warmup_base_model(lg, e, 4)
        assert [r.max_tokens for r in results] == [32]
        assert sorted(e.calls) == [32, 128, 512, 2048]  # All were in flight
        assert lg.error.called  # type: ignore[attr-defined]

    def test_adapter_steps_run_at_once(self, lg: Logger) -> None:
        baseline = [WarmupResult(max_tokens=t, finish_reason="stop") for t in (32, 128)]
        mgr = MagicMock()
        mgr.list.return_value = [_adapter("a1"), _adapter("a2")]
        e = _ConcurrentEngine(parties=2)
        warmup_adapters(lg, e, mgr, baseline, 4)
        assert len(e.calls) == 4
        assert e.peak == 2
        assert not lg.warning.called  # type: ignore[attr-defined]

    def test_simple_adapter_warmup_runs_at_once(self, lg: Logger) -> None:
        mgr = MagicMock()
        mgr.list.return_value = [_adapter("a1"), _adapter("a2"), _adapter("a3")]
        e = _ConcurrentEngine(parties=3)
        warmup_adapters(lg, e, mgr, [], 4)
        assert e.calls == [32, 32, 32]
        assert e.peak == 3
