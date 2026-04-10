"""Unit tests for BoundedQueueHandler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from appinfra.log import Logger

from llm_infer.serving.dispatch.handlers.bounded import (
    BoundedQueueHandler,
    RunningRequest,
)
from llm_infer.serving.dispatch.types import (
    RequestStatus,
    Response,
    StreamChunk,
)

from .._helpers import ResponseQueueFake as _ResponseQueue
from .._helpers import make_engine as _engine
from .._helpers import make_request as _request

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# submit()
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_accepts_under_capacity(self) -> None:
        h = BoundedQueueHandler(_engine(), max_pending=2)
        assert h.submit(_request("a")) is True
        assert h.submit(_request("b")) is True
        assert h.pending_count == 2

    def test_rejects_at_capacity(self) -> None:
        h = BoundedQueueHandler(_engine(), max_pending=1)
        assert h.submit(_request("a")) is True
        assert h.submit(_request("b")) is False
        assert h.pending_count == 1

    def test_is_saturated_property(self) -> None:
        h = BoundedQueueHandler(_engine(), max_pending=2)
        assert h.is_saturated is False
        h.submit(_request("a"))
        assert h.is_saturated is False
        h.submit(_request("b"))
        assert h.is_saturated is True


# ---------------------------------------------------------------------------
# pending_count counts queue, current, and running
# ---------------------------------------------------------------------------


class TestPendingCount:
    def test_empty(self) -> None:
        h = BoundedQueueHandler(_engine())
        assert h.pending_count == 0

    def test_queued_only(self) -> None:
        h = BoundedQueueHandler(_engine())
        h.submit(_request("a"))
        h.submit(_request("b"))
        assert h.pending_count == 2

    def test_includes_current(self) -> None:
        h = BoundedQueueHandler(_engine())
        h.current = _request("a")
        assert h.pending_count == 1

    def test_includes_running(self) -> None:
        h = BoundedQueueHandler(_engine())
        h.running["x"] = MagicMock()
        h.running["y"] = MagicMock()
        assert h.pending_count == 2


# ---------------------------------------------------------------------------
# engine property
# ---------------------------------------------------------------------------


def test_engine_property() -> None:
    e = _engine()
    h = BoundedQueueHandler(e)
    assert h.engine is e


# ---------------------------------------------------------------------------
# step() - single mode (max_batch_size=1)
# ---------------------------------------------------------------------------


class TestStepSingle:
    def test_step_empty_returns_empty(self) -> None:
        h = BoundedQueueHandler(_engine())
        assert h.step() == []

    def test_step_processes_one_request(self) -> None:
        e = _engine(generate_result="result")
        h = BoundedQueueHandler(e, max_pending=2)
        h.submit(_request("a"))
        results = h.step()
        assert len(results) == 1
        assert results[0].id == "a"
        assert results[0].status == RequestStatus.COMPLETED
        assert results[0].result == "result"

    def test_step_streaming_returns_empty(self) -> None:
        """Streaming requests don't return a Response - chunks go via response queue."""
        e = _engine()
        # generate_stream_sync returns a stream-like object
        stream = MagicMock()
        stream.__iter__ = lambda self: iter(["tok1", "tok2"])
        stream.finish_reason = "stop"
        stream.prompt_tokens = 5
        stream.completion_tokens = 2
        stream.adapter_info = None
        stream.adapter_mismatch = False
        e.generate_stream_sync.return_value = stream

        h = BoundedQueueHandler(e, max_pending=2)
        rq = _ResponseQueue()
        h.set_response_queue(rq)  # type: ignore[arg-type]
        h.submit(_request("a", stream=True))
        results = h.step()
        # Streaming response goes to queue, not returned
        assert results == []
        # Queue has token chunks + final
        assert any(isinstance(item, StreamChunk) for item in rq.items)
        assert any(getattr(item, "is_final", False) for item in rq.items)


# ---------------------------------------------------------------------------
# step_batched mode
# ---------------------------------------------------------------------------


def _make_batched_handler() -> tuple[BoundedQueueHandler, MagicMock]:
    """Create a handler in batched mode with a fully mocked engine."""
    e = MagicMock()
    e.tokenize.return_value = [1, 2, 3]
    e.build_stop_token_ids.return_value = set()
    e.decode_tokens.return_value = "decoded"
    e.count_tokens.return_value = 3
    h = BoundedQueueHandler(e, max_pending=10, max_batch_size=4)
    return h, e


class TestStepBatched:
    def test_empty_queue_returns_empty(self) -> None:
        h, _ = _make_batched_handler()
        assert h._step_batched() == []

    def test_streaming_request_uses_single_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A streaming request when batch_streaming=False routes to single mode."""
        h, e = _make_batched_handler()
        # Patch _process_request to avoid running real streaming logic
        h._process_request = MagicMock(
            return_value=Response(  # type: ignore[method-assign]
                id="r1", status=RequestStatus.COMPLETED, result="x"
            )
        )
        h.submit(_request("r1", stream=True))
        result = h._step_batched()
        assert result == []  # streaming returns empty

    def test_promotes_queued_to_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-streaming request gets promoted to running and decoded."""
        h, e = _make_batched_handler()
        # Mock _create_engine_request to return a fake EngineRequest-like object
        fake_engine_req = MagicMock()
        fake_engine_req.is_finished = False
        fake_engine_req.output_tokens = []
        fake_engine_req.prompt_tokens = [1, 2, 3]
        fake_engine_req.context = None
        h._create_engine_request = MagicMock(return_value=fake_engine_req)  # type: ignore[method-assign]
        h._run_prefill = MagicMock()  # type: ignore[method-assign]

        h.submit(_request("r1"))
        result = h._step_batched()
        # Engine.step_decode was called
        e.step_decode.assert_called_once()
        # Request is now running, no responses yet
        assert result == []
        assert "r1" in h.running

    def test_collects_finished_response(self) -> None:
        h, e = _make_batched_handler()
        # Pre-populate running with a finished request
        fake_eng = MagicMock()
        fake_eng.is_finished = True
        fake_eng.output_tokens = [4, 5, 6]
        fake_eng.prompt_tokens = [1, 2, 3]
        fake_eng.context = None
        h.running["r1"] = RunningRequest(
            request=_request("r1"),
            engine_request=fake_eng,
        )
        responses = h._step_batched()
        assert len(responses) == 1
        assert responses[0].status == RequestStatus.COMPLETED
        assert responses[0].id == "r1"
        assert "r1" not in h.running  # cleaned up
        e.free_request.assert_called_once()

    def test_step_routes_to_batched_mode(self) -> None:
        """step() dispatches to _step_batched when max_batch_size > 1."""
        h, _ = _make_batched_handler()
        result = h.step()  # empty queue -> []
        assert result == []


# ---------------------------------------------------------------------------
# _start_request error path
# ---------------------------------------------------------------------------


class TestStartRequestErrors:
    def test_start_failure_returns_none(self) -> None:
        h, e = _make_batched_handler()
        h._lg = MagicMock(spec=Logger)
        rq = _ResponseQueue()
        h.set_response_queue(rq)  # type: ignore[arg-type]
        e.tokenize.side_effect = RuntimeError("tokenizer broken")
        result = h._start_request(_request("r1"))
        assert result is None
        # Failure logged + reported
        h._lg.warning.assert_called()  # type: ignore[attr-defined]
        assert any(
            isinstance(item, Response) and item.status == RequestStatus.FAILED
            for item in rq.items
        )

    def test_start_failure_without_response_queue(self) -> None:
        """No queue set: failure is just logged, no response emitted."""
        h, e = _make_batched_handler()
        h._lg = MagicMock(spec=Logger)
        e.tokenize.side_effect = RuntimeError("boom")
        result = h._start_request(_request("r1"))
        assert result is None
        h._lg.warning.assert_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _stream_first_token / _stream_new_tokens
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_first_token_no_response_q(self) -> None:
        h, _ = _make_batched_handler()
        engine_req = MagicMock(output_tokens=[1])
        result = h._stream_first_token(_request("r1", stream=True), engine_req)
        assert result == (0, 0)

    def test_stream_first_token_not_streaming(self) -> None:
        h, _ = _make_batched_handler()
        h.set_response_queue(MagicMock())
        engine_req = MagicMock(output_tokens=[1])
        result = h._stream_first_token(_request("r1", stream=False), engine_req)
        assert result == (0, 0)

    def test_stream_first_token_no_output_tokens(self) -> None:
        h, _ = _make_batched_handler()
        h.set_response_queue(MagicMock())
        engine_req = MagicMock(output_tokens=[])
        result = h._stream_first_token(_request("r1", stream=True), engine_req)
        assert result == (0, 0)

    def test_stream_first_token_emits_chunk(self) -> None:
        h, e = _make_batched_handler()
        rq = _ResponseQueue()
        h.set_response_queue(rq)  # type: ignore[arg-type]
        engine_req = MagicMock(output_tokens=[42])
        e.decode_tokens.return_value = "hi"
        idx, length = h._stream_first_token(_request("r1", stream=True), engine_req)
        assert idx == 1
        assert length == 2
        assert len(rq.items) == 1
        assert rq.items[0].token == "hi"

    def test_stream_first_token_filters_replacement_chars(self) -> None:
        """Incomplete UTF-8 bytes (replacement char) don't advance position."""
        h, e = _make_batched_handler()
        rq = _ResponseQueue()
        h.set_response_queue(rq)  # type: ignore[arg-type]
        engine_req = MagicMock(output_tokens=[42])
        e.decode_tokens.return_value = "\ufffd"  # replacement char only
        idx, length = h._stream_first_token(_request("r1", stream=True), engine_req)
        assert idx == 1
        assert length == 0  # didn't advance
        assert rq.items == []

    def test_stream_new_tokens_no_response_q(self) -> None:
        h, _ = _make_batched_handler()
        # No queue set
        h._stream_new_tokens()  # Should not raise

    def test_stream_new_tokens_emits_clean_text(self) -> None:
        h, e = _make_batched_handler()
        rq = _ResponseQueue()
        h.set_response_queue(rq)  # type: ignore[arg-type]
        eng = MagicMock()
        eng.output_tokens = [1, 2]
        h.running["r1"] = RunningRequest(
            request=_request("r1", stream=True),
            engine_request=eng,
            last_streamed_idx=1,
            last_streamed_len=3,
        )
        e.decode_tokens.return_value = "abcdef"
        h._stream_new_tokens()
        # Decoded "abcdef", already streamed 3 chars -> emit "def"
        assert len(rq.items) == 1
        assert rq.items[0].token == "def"

    def test_stream_new_tokens_skips_non_streaming(self) -> None:
        h, e = _make_batched_handler()
        rq = _ResponseQueue()
        h.set_response_queue(rq)  # type: ignore[arg-type]
        eng = MagicMock()
        eng.output_tokens = [1, 2]
        h.running["r1"] = RunningRequest(
            request=_request("r1", stream=False),
            engine_request=eng,
        )
        h._stream_new_tokens()
        assert rq.items == []

    def test_stream_new_tokens_skips_when_no_new(self) -> None:
        h, e = _make_batched_handler()
        rq = _ResponseQueue()
        h.set_response_queue(rq)  # type: ignore[arg-type]
        eng = MagicMock()
        eng.output_tokens = [1, 2]
        h.running["r1"] = RunningRequest(
            request=_request("r1", stream=True),
            engine_request=eng,
            last_streamed_idx=2,  # Already streamed up to here
        )
        h._stream_new_tokens()
        assert rq.items == []


# ---------------------------------------------------------------------------
# _get_stream_finish_reason
# ---------------------------------------------------------------------------


class TestGetStreamFinishReason:
    def test_stop_token_returns_stop(self) -> None:
        h, _ = _make_batched_handler()
        eng = MagicMock()
        eng.finish_reason = None
        eng.output_tokens = [1, 2, 99]
        eng.stop_token_ids = {99}
        assert h._get_stream_finish_reason(eng) == "stop"

    def test_no_stop_token_returns_finish_reason(self) -> None:
        h, _ = _make_batched_handler()
        eng = MagicMock()
        eng.finish_reason = "length"
        eng.output_tokens = [1, 2, 3]
        eng.stop_token_ids = {99}
        assert h._get_stream_finish_reason(eng) == "length"

    def test_no_finish_reason_defaults_to_length(self) -> None:
        h, _ = _make_batched_handler()
        eng = MagicMock()
        eng.finish_reason = None
        eng.output_tokens = [1, 2]
        eng.stop_token_ids = set()
        assert h._get_stream_finish_reason(eng) == "length"


# ---------------------------------------------------------------------------
# _send_final_stream_chunk
# ---------------------------------------------------------------------------


def test_send_final_stream_chunk() -> None:
    h, e = _make_batched_handler()
    rq = _ResponseQueue()
    h.set_response_queue(rq)  # type: ignore[arg-type]
    eng = MagicMock()
    eng.context = None
    eng.finish_reason = "stop"
    eng.output_tokens = [1, 2, 3]
    eng.prompt_tokens = [4, 5]
    eng.stop_token_ids = set()
    running = RunningRequest(
        request=_request("r1", stream=True),
        engine_request=eng,
    )
    h._send_final_stream_chunk(running)
    assert len(rq.items) == 1
    chunk = rq.items[0]
    assert isinstance(chunk, StreamChunk)
    assert chunk.is_final is True
    assert chunk.finish_reason == "stop"
    assert chunk.prompt_tokens == 2
    assert chunk.completion_tokens == 3


# ---------------------------------------------------------------------------
# sequence_stats
# ---------------------------------------------------------------------------


class TestSequenceStats:
    def test_empty(self) -> None:
        h, _ = _make_batched_handler()
        assert h.sequence_stats() == {"active": 0, "total_tokens": 0}

    def test_with_running_requests(self) -> None:
        h, _ = _make_batched_handler()
        eng1 = MagicMock(prompt_tokens=[1, 2, 3], output_tokens=[4, 5])
        eng2 = MagicMock(prompt_tokens=[1], output_tokens=[2, 3, 4])
        h.running["r1"] = RunningRequest(request=_request("r1"), engine_request=eng1)
        h.running["r2"] = RunningRequest(request=_request("r2"), engine_request=eng2)
        stats = h.sequence_stats()
        assert stats == {"active": 2, "total_tokens": 9}
