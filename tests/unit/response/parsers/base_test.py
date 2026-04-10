"""Unit tests for response/parsers/base.py."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from llm_infer.response.events import EventType, StreamEvent
from llm_infer.response.parsers.base import BaseParser

pytestmark = pytest.mark.unit


class _PassthroughParser:
    """Test parser that yields each token as a TEXT event."""

    def __init__(self) -> None:
        self.feed_count = 0
        self.flush_count = 0
        self.reset_count = 0

    def feed(self, token: str) -> Iterator[StreamEvent]:
        self.feed_count += 1
        yield StreamEvent(EventType.TEXT, token)

    def flush(self) -> Iterator[StreamEvent]:
        self.flush_count += 1
        return iter([])

    def reset(self) -> None:
        self.reset_count += 1


class _UpperParser:
    """Test parser that uppercases TEXT content."""

    def feed(self, token: str) -> Iterator[StreamEvent]:
        yield StreamEvent(EventType.TEXT, token.upper())

    def flush(self) -> Iterator[StreamEvent]:
        return iter([])

    def reset(self) -> None:
        pass


class _ThinkEmittingParser:
    """Test parser that emits a non-TEXT event."""

    def feed(self, token: str) -> Iterator[StreamEvent]:
        yield StreamEvent(EventType.THINK_CONTENT, token)

    def flush(self) -> Iterator[StreamEvent]:
        yield StreamEvent(EventType.THINK_END, "")

    def reset(self) -> None:
        pass


class TestBaseParser:
    def test_empty_parsers_passthrough_text(self) -> None:
        parser = BaseParser()
        events = list(parser.feed("hello"))
        assert len(events) == 1
        assert events[0].type == EventType.TEXT
        assert events[0].content == "hello"

    def test_empty_parsers_flush_returns_nothing(self) -> None:
        parser = BaseParser()
        assert list(parser.flush()) == []

    def test_single_parser(self) -> None:
        p1 = _PassthroughParser()
        parser = BaseParser([p1])
        events = list(parser.feed("hello"))
        assert len(events) == 1
        assert p1.feed_count == 1

    def test_chain_parsers_text_flows_through(self) -> None:
        """TEXT events flow through chain, each parser modifies."""
        parser = BaseParser([_PassthroughParser(), _UpperParser()])
        events = list(parser.feed("hello"))
        assert len(events) == 1
        assert events[0].content == "HELLO"

    def test_non_text_event_skips_chain(self) -> None:
        """Non-TEXT events bypass downstream parsers."""
        parser = BaseParser([_ThinkEmittingParser(), _UpperParser()])
        events = list(parser.feed("reasoning"))
        # THINK_CONTENT bypasses upper parser
        assert events[0].type == EventType.THINK_CONTENT
        assert events[0].content == "reasoning"  # Not uppercased

    def test_flush_chains_text_events(self) -> None:
        parser = BaseParser([_ThinkEmittingParser(), _UpperParser()])
        events = list(parser.flush())
        # THINK_END from first, no flush from upper
        assert any(e.type == EventType.THINK_END for e in events)

    def test_reset_propagates(self) -> None:
        p1 = _PassthroughParser()
        p2 = _PassthroughParser()
        parser = BaseParser([p1, p2])
        parser.reset()
        assert p1.reset_count == 1
        assert p2.reset_count == 1
