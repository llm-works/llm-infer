"""Unit tests for ThinkTagParser."""

import pytest

from llm_infer.response import EventType, StreamEvent, ThinkTagParser

pytestmark = pytest.mark.unit


def collect_events(parser: ThinkTagParser, text: str) -> list[StreamEvent]:
    """Helper to collect all events from feeding text and flushing."""
    events = list(parser.feed(text))
    events.extend(parser.flush())
    return events


def stream_text(parser: ThinkTagParser, text: str) -> list[StreamEvent]:
    """Helper to stream text character by character."""
    events = []
    for char in text:
        events.extend(parser.feed(char))
    events.extend(parser.flush())
    return events


class TestThinkTagParserBasic:
    """Test basic think tag parsing."""

    def test_plain_text(self) -> None:
        """Test plain text without tags."""
        parser = ThinkTagParser()
        events = collect_events(parser, "hello world")
        assert len(events) == 1
        assert events[0] == StreamEvent(EventType.TEXT, "hello world")

    def test_simple_think_block(self) -> None:
        """Test simple think block detection."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<think>thinking</think>")
        assert len(events) == 3
        assert events[0].type == EventType.THINK_START
        assert events[1] == StreamEvent(EventType.THINK_CONTENT, "thinking")
        assert events[2].type == EventType.THINK_END

    def test_thinking_variant(self) -> None:
        """Test <thinking> variant tag."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<thinking>content</thinking>")
        assert len(events) == 3
        assert events[0].type == EventType.THINK_START
        assert events[1] == StreamEvent(EventType.THINK_CONTENT, "content")
        assert events[2].type == EventType.THINK_END

    def test_text_before_think(self) -> None:
        """Test text before think block."""
        parser = ThinkTagParser()
        events = collect_events(parser, "before<think>inside</think>")
        assert len(events) == 4
        assert events[0] == StreamEvent(EventType.TEXT, "before")
        assert events[1].type == EventType.THINK_START
        assert events[2] == StreamEvent(EventType.THINK_CONTENT, "inside")
        assert events[3].type == EventType.THINK_END

    def test_text_after_think(self) -> None:
        """Test text after think block."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<think>inside</think>after")
        assert len(events) == 4
        assert events[0].type == EventType.THINK_START
        assert events[1] == StreamEvent(EventType.THINK_CONTENT, "inside")
        assert events[2].type == EventType.THINK_END
        assert events[3] == StreamEvent(EventType.TEXT, "after")

    def test_multiple_think_blocks(self) -> None:
        """Test multiple think blocks."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<think>one</think>middle<think>two</think>")
        types = [e.type for e in events]
        assert types == [
            EventType.THINK_START,
            EventType.THINK_CONTENT,
            EventType.THINK_END,
            EventType.TEXT,
            EventType.THINK_START,
            EventType.THINK_CONTENT,
            EventType.THINK_END,
        ]


class TestThinkTagParserStreaming:
    """Test streaming edge cases."""

    def test_tag_split_across_chunks(self) -> None:
        """Test tag split across multiple chunks."""
        parser = ThinkTagParser()
        events = []
        events.extend(parser.feed("<thin"))
        events.extend(parser.feed("k>content</thi"))
        events.extend(parser.feed("nk>"))
        events.extend(parser.flush())

        types = [e.type for e in events]
        assert EventType.THINK_START in types
        assert EventType.THINK_CONTENT in types
        assert EventType.THINK_END in types

    def test_single_char_streaming(self) -> None:
        """Test character-by-character streaming."""
        parser = ThinkTagParser()
        events = stream_text(parser, "<think>hi</think>")

        types = [e.type for e in events]
        assert EventType.THINK_START in types
        assert EventType.THINK_END in types

        # Collect all content
        content = "".join(
            e.content for e in events if e.type == EventType.THINK_CONTENT
        )
        assert content == "hi"

    def test_incomplete_tag_at_end(self) -> None:
        """Test incomplete tag at end of stream."""
        parser = ThinkTagParser()
        events = collect_events(parser, "hello<thin")
        # Should emit the incomplete tag as text
        content = "".join(e.content for e in events if e.type == EventType.TEXT)
        assert content == "hello<thin"

    def test_unclosed_think_block(self) -> None:
        """Test unclosed think block flushes as content."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<think>unclosed content")
        types = [e.type for e in events]
        assert EventType.THINK_START in types
        assert EventType.THINK_CONTENT in types
        assert EventType.THINK_END in types  # Flush should close it

    def test_flush_empty_buffer(self) -> None:
        """Test flush with empty buffer."""
        parser = ThinkTagParser()
        events = list(parser.flush())
        assert events == []

    def test_consecutive_flushes(self) -> None:
        """Test multiple flushes are safe."""
        parser = ThinkTagParser()
        list(parser.feed("hello"))
        list(parser.flush())
        events = list(parser.flush())
        assert events == []


class TestThinkTagParserEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_input(self) -> None:
        """Test empty string input."""
        parser = ThinkTagParser()
        events = collect_events(parser, "")
        assert events == []

    def test_empty_think_block(self) -> None:
        """Test empty think block."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<think></think>")
        types = [e.type for e in events]
        assert types == [EventType.THINK_START, EventType.THINK_END]

    def test_nested_angle_brackets(self) -> None:
        """Test content with angle brackets that aren't tags."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<think>1 < 2 and 3 > 1</think>")
        content = "".join(
            e.content for e in events if e.type == EventType.THINK_CONTENT
        )
        assert content == "1 < 2 and 3 > 1"

    def test_unicode_content(self) -> None:
        """Test Unicode content inside tags."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<think>\u3053\u3093\u306b\u3061\u306f</think>")
        content = "".join(
            e.content for e in events if e.type == EventType.THINK_CONTENT
        )
        assert content == "\u3053\u3093\u306b\u3061\u306f"

    def test_newlines_in_content(self) -> None:
        """Test multiline content."""
        parser = ThinkTagParser()
        events = collect_events(parser, "<think>line1\nline2\nline3</think>")
        content = "".join(
            e.content for e in events if e.type == EventType.THINK_CONTENT
        )
        assert content == "line1\nline2\nline3"

    def test_custom_tags(self) -> None:
        """Test custom tag configuration."""
        parser = ThinkTagParser(
            open_tags=["<thought>"],
            close_tags=["</thought>"],
        )
        events = collect_events(parser, "<thought>custom</thought>")
        types = [e.type for e in events]
        assert EventType.THINK_START in types
        assert EventType.THINK_END in types


class TestThinkTagParserReset:
    """Test parser reset functionality."""

    def test_reset_clears_state(self) -> None:
        """Test that reset clears internal state."""
        parser = ThinkTagParser()
        # Start a think block but don't close it
        list(parser.feed("<think>partial"))
        # Reset
        parser.reset()
        # Should behave as fresh parser
        events = collect_events(parser, "fresh text")
        assert len(events) == 1
        assert events[0] == StreamEvent(EventType.TEXT, "fresh text")

    def test_reset_after_complete_stream(self) -> None:
        """Test reset after complete processing."""
        parser = ThinkTagParser()
        collect_events(parser, "<think>content</think>")
        parser.reset()
        events = collect_events(parser, "<think>new</think>")
        content = "".join(
            e.content for e in events if e.type == EventType.THINK_CONTENT
        )
        assert content == "new"
