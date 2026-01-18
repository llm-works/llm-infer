"""Unit tests for ResponseProcessor."""

from io import StringIO

import pytest

from llm_infer.response import (
    BaseResolver,
    EventType,
    ResponseProcessor,
    StreamEvent,
    TerminalResolver,
    ThinkTagParser,
)

pytestmark = pytest.mark.unit


class TestResponseProcessorDefaults:
    """Test default processor behavior."""

    def test_default_parser_is_think_tag_parser(self) -> None:
        """Test default parser is ThinkTagParser."""
        processor = ResponseProcessor()
        assert isinstance(processor.parser, ThinkTagParser)

    def test_default_resolver_is_terminal_resolver(self) -> None:
        """Test default resolver is TerminalResolver."""
        processor = ResponseProcessor()
        assert isinstance(processor.resolver, TerminalResolver)


class TestResponseProcessorCustomization:
    """Test processor customization."""

    def test_custom_resolver(self) -> None:
        """Test using custom resolver."""
        custom_resolver = BaseResolver()
        processor = ResponseProcessor(resolver=custom_resolver)
        assert processor.resolver is custom_resolver

    def test_custom_parser(self) -> None:
        """Test using custom parser."""
        custom_parser = ThinkTagParser(open_tags=["<custom>"], close_tags=["</custom>"])
        processor = ResponseProcessor(parser=custom_parser)
        assert processor.parser is custom_parser


class TestResponseProcessorFeed:
    """Test feed method."""

    def test_feed_processes_token(self) -> None:
        """Test feed processes tokens through parser and resolver."""
        output = StringIO()
        resolver = TerminalResolver(output=output)
        processor = ResponseProcessor(resolver=resolver)

        processor.feed("hello world")
        processor.finish()

        assert "hello world" in output.getvalue()

    def test_feed_handles_think_blocks(self) -> None:
        """Test feed handles think blocks."""
        output = StringIO()
        resolver = TerminalResolver(output=output)
        processor = ResponseProcessor(resolver=resolver)

        processor.feed("<think>thinking</think>")
        processor.finish()

        result = output.getvalue()
        # Should contain ANSI codes for styling
        assert "thinking" in result

    def test_feed_streaming(self) -> None:
        """Test feed works with streaming input."""
        output = StringIO()
        resolver = TerminalResolver(output=output)
        processor = ResponseProcessor(resolver=resolver)

        for char in "hello":
            processor.feed(char)
        processor.finish()

        assert "hello" in output.getvalue()


class TestResponseProcessorFinish:
    """Test finish method."""

    def test_finish_flushes_parser(self) -> None:
        """Test finish flushes parser buffer."""
        output = StringIO()
        resolver = TerminalResolver(output=output)
        processor = ResponseProcessor(resolver=resolver)

        # Feed partial content that would be buffered
        processor.feed("hello<thin")
        processor.finish()

        # Should have flushed the buffered content
        assert "hello<thin" in output.getvalue()

    def test_finish_calls_resolver_finish(self) -> None:
        """Test finish calls resolver.finish()."""
        finished = []

        class TrackingResolver(BaseResolver):
            def finish(self) -> None:
                finished.append(True)

        processor = ResponseProcessor(resolver=TrackingResolver())
        processor.finish()

        assert finished == [True]

    def test_finish_closes_unclosed_think_block(self) -> None:
        """Test finish resets ANSI styling for unclosed think block."""
        output = StringIO()
        resolver = TerminalResolver(output=output)
        processor = ResponseProcessor(resolver=resolver)

        # Feed unclosed think block
        processor.feed("<think>thinking content")
        processor.finish()

        result = output.getvalue()
        # Should contain the think content and reset ANSI codes
        assert "thinking content" in result
        # ANSI reset code should be present (from on_finish cleanup)
        assert "\x1b[0m" in result

    def test_finish_closes_unclosed_code_block(self) -> None:
        """Test finish emits closing fence for unclosed code block."""
        from llm_infer.response import CodeBlockParser
        from llm_infer.response.parsers.base import BaseParser

        output = StringIO()
        resolver = TerminalResolver(output=output)
        # Use CodeBlockParser to parse code fences
        parser = BaseParser([CodeBlockParser()])
        processor = ResponseProcessor(parser=parser, resolver=resolver)

        # Feed unclosed code block
        processor.feed("```python\nprint('hello')")
        processor.finish()

        result = output.getvalue()
        # Should contain opening fence, content, and closing fence
        assert "```python" in result
        assert "print('hello')" in result
        # Closing fence should be added by on_finish cleanup
        assert result.count("```") >= 2  # At least open and close


class TestResponseProcessorReset:
    """Test reset method."""

    def test_reset_clears_state(self) -> None:
        """Test reset clears both parser and resolver state."""
        output = StringIO()
        resolver = TerminalResolver(output=output)
        processor = ResponseProcessor(resolver=resolver)

        # Process some content
        processor.feed("<think>partial")
        processor.reset()

        # Feed new content - should be treated as fresh stream
        processor.feed("fresh text")
        processor.finish()

        # The output might have partial content from before reset,
        # but the parser state should be cleared
        # Reset should allow reprocessing
        assert "fresh text" in output.getvalue()


class TestResponseProcessorIntegration:
    """Integration tests for processor pipeline."""

    def test_full_response_processing(self) -> None:
        """Test processing a complete response."""
        output = StringIO()
        resolver = TerminalResolver(output=output)
        processor = ResponseProcessor(resolver=resolver)

        response = "Hello! <think>Let me think about this...</think> Here's my answer."
        processor.feed(response)
        processor.finish()

        result = output.getvalue()
        assert "Hello!" in result
        assert "Let me think about this..." in result
        assert "Here's my answer." in result

    def test_streaming_response_processing(self) -> None:
        """Test processing a streamed response."""
        output = StringIO()
        resolver = TerminalResolver(output=output)
        processor = ResponseProcessor(resolver=resolver)

        # Simulate streaming with chunks
        chunks = ["Hello", "! <thi", "nk>thinking", "</think>", " Done."]
        for chunk in chunks:
            processor.feed(chunk)
        processor.finish()

        result = output.getvalue()
        assert "Hello" in result
        assert "thinking" in result
        assert "Done" in result

    def test_collecting_resolver(self) -> None:
        """Test with a resolver that collects events."""
        events: list[StreamEvent] = []

        class CollectingResolver(BaseResolver):
            def handle(self, event: StreamEvent) -> None:
                events.append(event)
                super().handle(event)

        processor = ResponseProcessor(resolver=CollectingResolver())
        processor.feed("<think>test</think>")
        processor.finish()

        types = [e.type for e in events]
        assert EventType.THINK_START in types
        assert EventType.THINK_CONTENT in types
        assert EventType.THINK_END in types


class TestResponseProcessorProtocolCompliance:
    """Test that custom implementations work with protocols."""

    def test_custom_parser_protocol(self) -> None:
        """Test custom parser implementing Parser protocol."""
        from collections.abc import Iterator

        class PassthroughParser:
            def feed(self, token: str) -> Iterator[StreamEvent]:
                yield StreamEvent(EventType.TEXT, token)

            def flush(self) -> Iterator[StreamEvent]:
                return iter([])

            def reset(self) -> None:
                pass

        output = StringIO()
        processor = ResponseProcessor(
            parser=PassthroughParser(),
            resolver=TerminalResolver(output=output),
        )

        processor.feed("hello")
        processor.finish()

        assert "hello" in output.getvalue()

    def test_custom_resolver_protocol(self) -> None:
        """Test custom resolver implementing Resolver protocol."""
        collected: list[str] = []

        class TextCollector:
            def handle(self, event: StreamEvent) -> None:
                if event.type == EventType.TEXT:
                    collected.append(event.content)

            def finish(self) -> None:
                pass

            def reset(self) -> None:
                collected.clear()

        processor = ResponseProcessor(resolver=TextCollector())
        processor.feed("hello world")
        processor.finish()

        assert "".join(collected) == "hello world"
