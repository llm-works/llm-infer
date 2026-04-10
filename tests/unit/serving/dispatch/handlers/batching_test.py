"""Unit tests for ContinuousBatchingHandler (placeholder/stub handler)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.handlers.batching import (
    ContinuousBatchingHandler,
    RunningRequest,
)
from llm_infer.serving.dispatch.types import Request, RequestStatus

pytestmark = pytest.mark.unit


def _engine(*, generate_result: str = "out") -> MagicMock:
    e = MagicMock()
    e.generate.return_value = generate_result
    e.count_tokens.return_value = 5
    return e


def _request(req_id: str = "r1") -> Request:
    return Request(id=req_id, prompt="hello")


# ---------------------------------------------------------------------------
# RunningRequest
# ---------------------------------------------------------------------------


def test_running_request_decode_output() -> None:
    rr = RunningRequest(request=_request(), output_tokens=[1, 2, 3])
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"
    assert rr.get_output(tokenizer) == "decoded"
    tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_accepts_under_capacity(self) -> None:
        h = ContinuousBatchingHandler(_engine(), max_pending=3)
        assert h.submit(_request("a"))
        assert h.submit(_request("b"))
        assert h.pending_count == 2

    def test_rejects_at_capacity(self) -> None:
        h = ContinuousBatchingHandler(_engine(), max_pending=1)
        assert h.submit(_request("a"))
        assert not h.submit(_request("b"))


def test_engine_property() -> None:
    e = _engine()
    h = ContinuousBatchingHandler(e)
    assert h.engine is e


# ---------------------------------------------------------------------------
# pending_count / is_saturated
# ---------------------------------------------------------------------------


class TestProperties:
    def test_empty_pending(self) -> None:
        h = ContinuousBatchingHandler(_engine())
        assert h.pending_count == 0
        assert h.is_saturated is False

    def test_pending_counts_waiting_plus_running(self) -> None:
        h = ContinuousBatchingHandler(_engine(), max_pending=10)
        h.waiting.append(_request("a"))
        h.running["b"] = RunningRequest(request=_request("b"))
        assert h.pending_count == 2

    def test_is_saturated(self) -> None:
        h = ContinuousBatchingHandler(_engine(), max_pending=2)
        h.submit(_request("a"))
        h.submit(_request("b"))
        assert h.is_saturated is True


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


class TestStep:
    def test_empty_returns_empty(self) -> None:
        h = ContinuousBatchingHandler(_engine())
        assert h.step() == []

    def test_processes_pending(self) -> None:
        h = ContinuousBatchingHandler(_engine(generate_result="hello"))
        h.submit(_request("r1"))
        results = h.step()
        assert len(results) == 1
        assert results[0].id == "r1"
        assert results[0].status == RequestStatus.COMPLETED
        assert results[0].result == "hello"
        assert "r1" not in h.running

    def test_processes_multiple_in_one_step(self) -> None:
        h = ContinuousBatchingHandler(_engine())
        h.submit(_request("r1"))
        h.submit(_request("r2"))
        results = h.step()
        assert len(results) == 2

    def test_engine_failure_returns_failed_response(self) -> None:
        e = _engine()
        e.generate.side_effect = RuntimeError("boom")
        h = ContinuousBatchingHandler(e)
        h.submit(_request("r1"))
        results = h.step()
        assert len(results) == 1
        assert results[0].status == RequestStatus.FAILED
        assert "boom" in results[0].error

    def test_respects_max_batch_size(self) -> None:
        """Only max_batch_size requests get promoted to running per step."""
        h = ContinuousBatchingHandler(_engine(), max_batch_size=2, max_pending=10)
        h.submit(_request("r1"))
        h.submit(_request("r2"))
        h.submit(_request("r3"))
        # First step: 2 promoted + processed; r3 still waiting
        results = h.step()
        assert len(results) == 2
        assert len(h.waiting) == 1
        # Next step processes the remaining one
        results = h.step()
        assert len(results) == 1
