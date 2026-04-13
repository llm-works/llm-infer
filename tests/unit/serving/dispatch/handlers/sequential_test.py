"""Unit tests for SequentialHandler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_infer.serving.dispatch.handlers import SequentialHandler
from llm_infer.serving.dispatch.types import RequestStatus

from .._helpers import make_engine as _engine
from .._helpers import make_request as _request

pytestmark = pytest.mark.unit


class TestSubmit:
    def test_always_accepts(self) -> None:
        h = SequentialHandler(_engine())
        for i in range(100):
            assert h.submit(_request(f"r{i}")) is True
        assert h.pending_count == 100

    def test_is_saturated_always_false(self) -> None:
        h = SequentialHandler(_engine())
        h.submit(_request())
        assert h.is_saturated is False


def test_engine_property() -> None:
    e = _engine()
    h = SequentialHandler(e)
    assert h.engine is e


class TestStep:
    def test_empty_returns_empty(self) -> None:
        h = SequentialHandler(_engine())
        assert h.step() == []

    def test_processes_one_request(self) -> None:
        h = SequentialHandler(_engine())
        h.submit(_request("r1"))
        results = h.step()
        assert len(results) == 1
        assert results[0].id == "r1"
        assert results[0].status == RequestStatus.COMPLETED

    def test_processes_one_at_a_time(self) -> None:
        h = SequentialHandler(_engine())
        h.submit(_request("r1"))
        h.submit(_request("r2"))
        first = h.step()
        assert first[0].id == "r1"
        assert h.pending_count == 1
        second = h.step()
        assert second[0].id == "r2"

    def test_streaming_returns_empty(self) -> None:
        """Streaming requests put chunks on the queue, return [] from step."""
        h = SequentialHandler(_engine())
        rq = MagicMock()
        h.set_response_queue(rq)

        # Set up generate_stream_sync to return a stream
        stream = MagicMock()
        stream.__iter__ = lambda self: iter(["tok1"])
        stream.finish_reason = "stop"
        stream.prompt_tokens = 5
        stream.completion_tokens = 1
        stream.tool_calls = None
        stream.adapter_info = None
        stream.adapter_mismatch = False
        h._engine.generate_stream_sync.return_value = stream

        h.submit(_request("r1", stream=True))
        results = h.step()
        assert results == []  # Streaming returns empty


class TestPendingCount:
    def test_with_current(self) -> None:
        h = SequentialHandler(_engine())
        h.current = _request()
        assert h.pending_count == 1

    def test_with_queue_and_current(self) -> None:
        h = SequentialHandler(_engine())
        h.current = _request("a")
        h.queue.append(_request("b"))
        h.queue.append(_request("c"))
        assert h.pending_count == 3
