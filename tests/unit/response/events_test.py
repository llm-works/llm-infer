"""Unit tests for StreamEvent and EventType."""

import pytest

from llm_infer.response import EventType, StreamEvent

pytestmark = pytest.mark.unit


class TestEventType:
    """Test EventType enum."""

    def test_all_event_types_defined(self) -> None:
        """Test that all expected event types are defined."""
        expected = {
            "TEXT",
            "THINK_START",
            "THINK_CONTENT",
            "THINK_END",
            "CODE_START",
            "CODE_CONTENT",
            "CODE_END",
        }
        actual = {e.name for e in EventType}
        assert actual == expected


class TestStreamEvent:
    """Test StreamEvent dataclass."""

    def test_create_text_event(self) -> None:
        """Test creating a simple text event."""
        event = StreamEvent(EventType.TEXT, "hello")
        assert event.type == EventType.TEXT
        assert event.content == "hello"
        assert event.metadata == {}

    def test_create_event_with_metadata(self) -> None:
        """Test creating an event with metadata."""
        event = StreamEvent(EventType.CODE_START, "", {"language": "python"})
        assert event.type == EventType.CODE_START
        assert event.content == ""
        assert event.metadata == {"language": "python"}

    def test_event_is_immutable(self) -> None:
        """Test that events are frozen (immutable)."""
        event = StreamEvent(EventType.TEXT, "hello")
        with pytest.raises(AttributeError):
            event.content = "world"  # type: ignore[misc]

    def test_default_content(self) -> None:
        """Test default content is empty string."""
        event = StreamEvent(EventType.THINK_START)
        assert event.content == ""

    def test_default_metadata(self) -> None:
        """Test default metadata is empty dict."""
        event = StreamEvent(EventType.TEXT, "hello")
        assert event.metadata == {}

    def test_events_not_hashable_with_metadata(self) -> None:
        """Test that events with metadata are not hashable (dict is mutable)."""
        # Events without metadata default have empty dict which is unhashable
        event = StreamEvent(EventType.TEXT, "hello")
        with pytest.raises(TypeError):
            hash(event)

    def test_events_equality(self) -> None:
        """Test event equality."""
        event1 = StreamEvent(EventType.TEXT, "hello")
        event2 = StreamEvent(EventType.TEXT, "hello")
        event3 = StreamEvent(EventType.TEXT, "world")
        assert event1 == event2
        assert event1 != event3
