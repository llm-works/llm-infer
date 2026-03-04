"""Unit tests for request context."""

from unittest.mock import MagicMock

import pytest

from llm_infer.context import _DEBUG_EVENTS, Event, RequestContext

pytestmark = pytest.mark.unit


class TestEvent:
    """Test Event enum."""

    def test_debug_events(self) -> None:
        """Test that debug events are correctly categorized."""
        assert Event.REQUESTED in _DEBUG_EVENTS
        assert Event.TOKENIZED in _DEBUG_EVENTS
        assert Event.PREFILLED in _DEBUG_EVENTS
        assert Event.COMPLETE in _DEBUG_EVENTS

    def test_trace_events(self) -> None:
        """Test that trace events are not in debug set."""
        assert Event.DECODED not in _DEBUG_EVENTS
        assert Event.DECODE not in _DEBUG_EVENTS
        assert Event.KV_ALLOC not in _DEBUG_EVENTS
        assert Event.KV_FREE not in _DEBUG_EVENTS
        assert Event.SAMPLED not in _DEBUG_EVENTS

    def test_event_values(self) -> None:
        """Test event string values."""
        assert Event.REQUESTED.value == "requested"
        assert Event.COMPLETE.value == "complete"
        assert Event.DECODE.value == "decode"


class TestRequestContext:
    """Test RequestContext dataclass."""

    def test_creates_with_id(self) -> None:
        """Test context is created with given ID."""
        lg = MagicMock()
        ctx = RequestContext(id="test-123", lg=lg)
        assert ctx.id == "test-123"

    def test_start_time_initialized(self) -> None:
        """Test that start_time is initialized."""
        lg = MagicMock()
        ctx = RequestContext(id="test", lg=lg)
        assert ctx.start_time > 0

    def test_timings_empty_initially(self) -> None:
        """Test that timings dict is empty initially."""
        lg = MagicMock()
        ctx = RequestContext(id="test", lg=lg)
        assert ctx.timings == {}

    def test_mark_records_timing(self) -> None:
        """Test that mark records timing."""
        lg = MagicMock()
        ctx = RequestContext(id="test", lg=lg)

        ctx.mark(Event.REQUESTED)

        assert "requested" in ctx.timings
        assert ctx.timings["requested"] >= 0

    def test_mark_debug_event_calls_debug(self) -> None:
        """Test that debug events call lg.debug."""
        lg = MagicMock()
        ctx = RequestContext(id="test", lg=lg)

        ctx.mark(Event.REQUESTED)

        lg.debug.assert_called_once()

    def test_mark_trace_event_calls_trace(self) -> None:
        """Test that trace events call lg.trace."""
        lg = MagicMock()
        ctx = RequestContext(id="test", lg=lg)

        ctx.mark(Event.DECODE)

        lg.trace.assert_called_once()

    def test_mark_with_extra_data(self) -> None:
        """Test that mark passes extra data to logger."""
        lg = MagicMock()
        ctx = RequestContext(id="test", lg=lg)

        ctx.mark(Event.PREFILLED, tokens=100)

        lg.debug.assert_called_once()
        call_args = lg.debug.call_args
        assert call_args[1]["extra"]["tokens"] == 100

    def test_get_timings_csv(self) -> None:
        """Test CSV export of timings."""
        lg = MagicMock()
        ctx = RequestContext(id="test", lg=lg)

        ctx.mark(Event.REQUESTED)
        ctx.mark(Event.COMPLETE)

        csv = ctx.get_timings_csv()
        assert "requested=" in csv
        assert "complete=" in csv

    def test_multiple_marks_accumulate(self) -> None:
        """Test that multiple marks accumulate in timings."""
        lg = MagicMock()
        ctx = RequestContext(id="test", lg=lg)

        ctx.mark(Event.REQUESTED)
        ctx.mark(Event.TOKENIZED)
        ctx.mark(Event.PREFILLED)

        assert len(ctx.timings) == 3
