"""Unit tests for CodeBlockParser."""

import pytest

from llm_infer.response import CodeBlockParser, EventType, StreamEvent

pytestmark = pytest.mark.unit


def collect_events(parser: CodeBlockParser, text: str) -> list[StreamEvent]:
    """Helper to collect all events from feeding text and flushing."""
    events = list(parser.feed(text))
    events.extend(parser.flush())
    return events


def stream_text(parser: CodeBlockParser, text: str) -> list[StreamEvent]:
    """Helper to stream text character by character."""
    events = []
    for char in text:
        events.extend(parser.feed(char))
    events.extend(parser.flush())
    return events


class TestCodeBlockParserBasic:
    """Test basic code block parsing."""

    def test_plain_text(self) -> None:
        """Test plain text without code blocks."""
        parser = CodeBlockParser()
        events = collect_events(parser, "hello world")
        assert len(events) == 1
        assert events[0] == StreamEvent(EventType.TEXT, "hello world")

    def test_simple_code_block(self) -> None:
        """Test simple code block detection."""
        parser = CodeBlockParser()
        events = collect_events(parser, "```\ncode\n```")
        types = [e.type for e in events]
        assert EventType.CODE_START in types
        assert EventType.CODE_CONTENT in types
        assert EventType.CODE_END in types

    def test_code_block_with_language(self) -> None:
        """Test code block with language identifier."""
        parser = CodeBlockParser()
        events = collect_events(parser, "```python\nprint('hello')\n```")
        # Find CODE_START event
        start_event = next(e for e in events if e.type == EventType.CODE_START)
        assert start_event.metadata.get("language") == "python"

    def test_text_before_code(self) -> None:
        """Test text before code block."""
        parser = CodeBlockParser()
        events = collect_events(parser, "before\n```\ncode\n```")
        types = [e.type for e in events]
        assert types[0] == EventType.TEXT
        assert EventType.CODE_START in types

    def test_text_after_code(self) -> None:
        """Test text after code block."""
        parser = CodeBlockParser()
        events = collect_events(parser, "```\ncode\n```\nafter")
        types = [e.type for e in events]
        assert EventType.CODE_END in types
        assert types[-1] == EventType.TEXT

    def test_multiple_code_blocks(self) -> None:
        """Test multiple code blocks."""
        parser = CodeBlockParser()
        events = collect_events(parser, "```\none\n```\nmiddle\n```\ntwo\n```")
        # Count code block events
        starts = sum(1 for e in events if e.type == EventType.CODE_START)
        ends = sum(1 for e in events if e.type == EventType.CODE_END)
        assert starts == 2
        assert ends == 2


class TestCodeBlockParserStreaming:
    """Test streaming edge cases."""

    def test_fence_split_across_chunks(self) -> None:
        """Test fence split across multiple chunks."""
        parser = CodeBlockParser()
        events = []
        events.extend(parser.feed("``"))
        events.extend(parser.feed("`python\ncode"))
        events.extend(parser.feed("\n``"))
        events.extend(parser.feed("`"))
        events.extend(parser.flush())

        types = [e.type for e in events]
        assert EventType.CODE_START in types
        assert EventType.CODE_END in types

    def test_single_char_streaming(self) -> None:
        """Test character-by-character streaming."""
        parser = CodeBlockParser()
        events = stream_text(parser, "```\nhi\n```")

        types = [e.type for e in events]
        assert EventType.CODE_START in types
        assert EventType.CODE_END in types

    def test_incomplete_fence_at_end(self) -> None:
        """Test incomplete fence at end of stream."""
        parser = CodeBlockParser()
        events = collect_events(parser, "hello``")
        # Should emit the incomplete fence as text
        content = "".join(e.content for e in events if e.type == EventType.TEXT)
        assert "``" in content

    def test_unclosed_code_block(self) -> None:
        """Test unclosed code block flushes as content."""
        parser = CodeBlockParser()
        events = collect_events(parser, "```python\nunclosed code")
        types = [e.type for e in events]
        assert EventType.CODE_START in types
        assert EventType.CODE_CONTENT in types
        assert EventType.CODE_END in types  # Flush should close it


class TestCodeBlockParserEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_input(self) -> None:
        """Test empty string input."""
        parser = CodeBlockParser()
        events = collect_events(parser, "")
        assert events == []

    def test_empty_code_block(self) -> None:
        """Test empty code block."""
        parser = CodeBlockParser()
        events = collect_events(parser, "```\n```")
        types = [e.type for e in events]
        assert EventType.CODE_START in types
        assert EventType.CODE_END in types

    def test_backticks_in_text(self) -> None:
        """Test backticks that aren't code fences."""
        parser = CodeBlockParser()
        events = collect_events(parser, "use `inline` code")
        # Should be text, not code blocks
        types = [e.type for e in events]
        assert EventType.CODE_START not in types
        content = "".join(e.content for e in events)
        assert "`inline`" in content

    def test_various_languages(self) -> None:
        """Test various language identifiers."""
        languages = ["python", "javascript", "rust", "go", "cpp"]
        for lang in languages:
            parser = CodeBlockParser()
            events = collect_events(parser, f"```{lang}\ncode\n```")
            start_event = next(e for e in events if e.type == EventType.CODE_START)
            assert start_event.metadata.get("language") == lang


class TestCodeBlockParserReset:
    """Test parser reset functionality."""

    def test_reset_clears_state(self) -> None:
        """Test that reset clears internal state."""
        parser = CodeBlockParser()
        # Start a code block but don't close it
        list(parser.feed("```python\npartial"))
        # Reset
        parser.reset()
        # Should behave as fresh parser
        events = collect_events(parser, "fresh text")
        assert len(events) == 1
        assert events[0] == StreamEvent(EventType.TEXT, "fresh text")
