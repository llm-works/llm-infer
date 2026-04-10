"""Unit tests for serving/dispatch/warmup.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.dispatch.warmup import (
    WarmupResult,
    _build_token_sweep,
    _check_adapter_fallback,
    _extract_finish_reason,
    _get_max_model_len,
    warmup_adapters,
    warmup_base_model,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def lg() -> Logger:
    return MagicMock(spec=Logger)


def _engine(
    *,
    max_model_len: int = 4096,
    supports_embeddings: bool = False,
    generate_outputs: list[Any] | None = None,
    embed_result: Any = None,
    generate_side_effect: Exception | None = None,
) -> MagicMock:
    """Build a mock engine."""
    e = MagicMock()
    e.max_model_len = max_model_len
    e.supports_embeddings = lambda: supports_embeddings
    if generate_side_effect is not None:
        e.generate.side_effect = generate_side_effect
    elif generate_outputs is not None:
        e.generate.side_effect = generate_outputs
    else:
        e.generate.return_value = {"finish_reason": "stop"}
    if embed_result is not None:
        e.embed.return_value = embed_result
    return e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_get_max_model_len_default(self) -> None:
        e = MagicMock(spec=[])  # No attributes
        assert _get_max_model_len(e) == 4096

    def test_get_max_model_len_explicit(self) -> None:
        e = MagicMock()
        e.max_model_len = 8192
        assert _get_max_model_len(e) == 8192

    def test_get_max_model_len_none(self) -> None:
        e = MagicMock()
        e.max_model_len = None
        assert _get_max_model_len(e) == 4096

    def test_build_token_sweep_4096(self) -> None:
        # cap = 2048; sweep = [32, 128, 512, 2048]
        sweep = _build_token_sweep(4096)
        assert sweep == [32, 128, 512, 2048]

    def test_build_token_sweep_small(self) -> None:
        # cap = 256; sweep = [32, 128]
        sweep = _build_token_sweep(512)
        assert sweep == [32, 128]

    def test_build_token_sweep_tiny(self) -> None:
        # cap = 16; nothing fits
        sweep = _build_token_sweep(32)
        assert sweep == []

    def test_extract_finish_reason_dict(self) -> None:
        assert _extract_finish_reason({"finish_reason": "stop"}) == "stop"
        assert _extract_finish_reason({"finish_reason": "length"}) == "length"

    def test_extract_finish_reason_dict_missing(self) -> None:
        assert _extract_finish_reason({}) == "unknown"

    def test_extract_finish_reason_string(self) -> None:
        assert _extract_finish_reason("just a string") == "unknown"

    def test_check_adapter_fallback_true(self) -> None:
        assert _check_adapter_fallback({"adapter_info": {"fallback": True}}) is True

    def test_check_adapter_fallback_false(self) -> None:
        assert _check_adapter_fallback({"adapter_info": {"fallback": False}}) is False

    def test_check_adapter_fallback_no_adapter_info(self) -> None:
        assert _check_adapter_fallback({}) is False

    def test_check_adapter_fallback_string(self) -> None:
        assert _check_adapter_fallback("plain") is False

    def test_check_adapter_fallback_invalid_adapter_info(self) -> None:
        assert _check_adapter_fallback({"adapter_info": "not a dict"}) is False


# ---------------------------------------------------------------------------
# warmup_base_model
# ---------------------------------------------------------------------------


class TestWarmupBaseModel:
    def test_embedding_engine_only_runs_embed(self, lg: Logger) -> None:
        e = _engine(supports_embeddings=True, embed_result=[[0.1, 0.2]])
        results = warmup_base_model(lg, e)
        assert results == []
        e.embed.assert_called_once_with(["warmup"])

    def test_token_sweep_all_succeed(self, lg: Logger) -> None:
        e = _engine(max_model_len=4096)
        results = warmup_base_model(lg, e)
        assert len(results) == 4
        assert all(r.finish_reason == "stop" for r in results)
        assert [r.max_tokens for r in results] == [32, 128, 512, 2048]

    def test_step_failure_breaks_sweep(self, lg: Logger) -> None:
        # First call succeeds, second raises
        e = _engine(
            max_model_len=4096,
            generate_outputs=[
                {"finish_reason": "stop"},
                RuntimeError("boom"),
            ],
        )
        results = warmup_base_model(lg, e)
        # First step succeeded; second failed -> stops sweep
        assert len(results) == 1
        assert lg.error.called  # type: ignore[attr-defined]

    def test_small_max_model_len(self, lg: Logger) -> None:
        e = _engine(max_model_len=512)
        results = warmup_base_model(lg, e)
        # cap = 256, sweep = [32, 128]
        assert len(results) == 2


# ---------------------------------------------------------------------------
# warmup_adapters
# ---------------------------------------------------------------------------


def _adapter(key: str = "ad-1") -> MagicMock:
    a = MagicMock()
    a.key = key
    return a


class TestWarmupAdapters:
    def test_no_adapter_manager(self, lg: Logger) -> None:
        e = _engine()
        warmup_adapters(lg, e, None, [])
        e.generate.assert_not_called()

    def test_empty_adapter_list(self, lg: Logger) -> None:
        mgr = MagicMock()
        mgr.list.return_value = []
        e = _engine()
        warmup_adapters(lg, e, mgr, [])
        e.generate.assert_not_called()

    def test_no_baseline_runs_simple_warmup(self, lg: Logger) -> None:
        mgr = MagicMock()
        mgr.list.return_value = [_adapter("a1"), _adapter("a2")]
        e = _engine()
        warmup_adapters(lg, e, mgr, [])  # empty baseline
        # Each adapter ran a single warmup test
        assert e.generate.call_count == 2

    def test_simple_warmup_handles_exception(self, lg: Logger) -> None:
        mgr = MagicMock()
        mgr.list.return_value = [_adapter("a1")]
        e = _engine(generate_side_effect=RuntimeError("boom"))
        warmup_adapters(lg, e, mgr, [])
        assert lg.error.called  # type: ignore[attr-defined]

    def test_baseline_all_match(self, lg: Logger) -> None:
        baseline = [
            WarmupResult(max_tokens=32, finish_reason="stop"),
            WarmupResult(max_tokens=128, finish_reason="stop"),
        ]
        mgr = MagicMock()
        mgr.list.return_value = [_adapter("a1")]
        e = _engine(
            generate_outputs=[
                {"finish_reason": "stop"},
                {"finish_reason": "stop"},
            ]
        )
        warmup_adapters(lg, e, mgr, baseline)
        # No warning emitted
        warning_msgs = [
            c.args[0]
            for c in lg.warning.call_args_list  # type: ignore[attr-defined]
        ]
        assert "adapter warmup completed with issues" not in warning_msgs

    def test_baseline_eos_mismatch_warns(self, lg: Logger) -> None:
        """Base produces stop, adapter produces length -> EOS mismatch."""
        baseline = [WarmupResult(max_tokens=32, finish_reason="stop")]
        mgr = MagicMock()
        mgr.list.return_value = [_adapter("a1")]
        e = _engine(generate_outputs=[{"finish_reason": "length"}])
        warmup_adapters(lg, e, mgr, baseline)
        warning_msgs = [
            c.args[0]
            for c in lg.warning.call_args_list  # type: ignore[attr-defined]
        ]
        assert any("did not produce EOS" in m for m in warning_msgs)
        assert any("completed with issues" in m for m in warning_msgs)

    def test_adapter_fallback_detected(self, lg: Logger) -> None:
        """vLLM falling back to base model is logged as warning and counted as failure."""
        baseline = [WarmupResult(max_tokens=32, finish_reason="stop")]
        mgr = MagicMock()
        mgr.list.return_value = [_adapter("a1")]
        e = _engine(
            generate_outputs=[
                {"finish_reason": "stop", "adapter_info": {"fallback": True}}
            ]
        )
        warmup_adapters(lg, e, mgr, baseline)
        warning_msgs = [
            c.args[0]
            for c in lg.warning.call_args_list  # type: ignore[attr-defined]
        ]
        assert any("fell back to base model" in m for m in warning_msgs)

    def test_adapter_exception_during_test(self, lg: Logger) -> None:
        baseline = [WarmupResult(max_tokens=32, finish_reason="stop")]
        mgr = MagicMock()
        mgr.list.return_value = [_adapter("a1")]
        e = _engine(generate_side_effect=RuntimeError("boom"))
        warmup_adapters(lg, e, mgr, baseline)
        assert lg.error.called  # type: ignore[attr-defined]
