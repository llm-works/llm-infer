"""Unit tests for serving/dispatch/progress.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.dispatch.progress import ProgressTracker

pytestmark = pytest.mark.unit


@pytest.fixture
def lg() -> Logger:
    return MagicMock(spec=Logger)


def _info_messages(lg: Logger) -> list[str]:
    return [c.args[0] for c in lg.info.call_args_list if c.args]  # type: ignore[attr-defined]


def _debug_messages(lg: Logger) -> list[str]:
    return [c.args[0] for c in lg.debug.call_args_list if c.args]  # type: ignore[attr-defined]


class TestProgressTracker:
    def test_phase_start_logs_initializing(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("weights:init", 0, 100)
        msgs = _debug_messages(lg)
        assert any("initializing weights" in m for m in msgs)

    def test_phase_complete_logs_completed(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("weights:init", 100, 100)
        msgs = _info_messages(lg)
        assert any("weights initialized" in m for m in msgs)

    def test_progress_emits_debug_at_steps(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("weights:stream", 0, 100)
        lg.debug.reset_mock()  # type: ignore[attr-defined]
        # 10% step boundary
        t("weights:stream", 10, 100)
        msgs = _debug_messages(lg)
        assert any("loading weights" in m for m in msgs)

    def test_progress_skips_under_step_threshold(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("weights:stream", 0, 100)
        lg.debug.reset_mock()  # type: ignore[attr-defined]
        # 1% — below 10% step threshold
        t("weights:stream", 1, 100)
        msgs = _debug_messages(lg)
        assert not any("loading weights" in m for m in msgs)

    def test_unknown_phase_uses_default_labels(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("custom-phase", 0, 100)
        t("custom-phase", 100, 100)
        info = _info_messages(lg)
        assert any("custom-phase loaded" in m for m in info)

    def test_kv_cache_phase(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("kv_cache", 100, 100)
        info = _info_messages(lg)
        assert any("kv_cache loaded" in m for m in info)

    def test_tokenizer_phase(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("tokenizer", 1, 1)
        info = _info_messages(lg)
        assert any("tokenizer loaded" in m for m in info)

    def test_weights_alloc_phase(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("weights:alloc", 100, 100)
        info = _info_messages(lg)
        assert any("weights allocated" in m for m in info)

    def test_progress_includes_eta(self, lg: Logger) -> None:
        t = ProgressTracker(lg)
        t("weights:stream", 0, 1000)
        lg.debug.reset_mock()  # type: ignore[attr-defined]
        t("weights:stream", 100, 1000)
        progress_calls = [
            c
            for c in lg.debug.call_args_list  # type: ignore[attr-defined]
            if c.args and "loading weights" in c.args[0]
        ]
        assert len(progress_calls) >= 1
        extra = progress_calls[0].kwargs.get("extra", {})
        assert "progress" in extra
        assert extra["progress"] == "10%"

    def test_total_one_no_eta(self, lg: Logger) -> None:
        """When total <= 1, no ETA is created (avoids div-by-zero)."""
        t = ProgressTracker(lg)
        t("tokenizer", 0, 1)
        t("tokenizer", 1, 1)

    def test_zero_progress_is_phase_start_only(self, lg: Logger) -> None:
        """current=0 only initializes the phase, doesn't emit a progress update.

        Phase init logs 'loading weights...' (no extra dict).
        Progress updates also use 'loading weights...' but with an extra dict.
        """
        t = ProgressTracker(lg)
        t("weights:stream", 0, 100)
        progress_calls = [
            c
            for c in lg.debug.call_args_list  # type: ignore[attr-defined]
            if c.args and "loading weights" in c.args[0] and c.kwargs.get("extra")
        ]
        # Only the phase-start log (no extra dict), no progress logs
        assert progress_calls == []
