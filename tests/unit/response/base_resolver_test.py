"""Unit tests for BaseResolver."""

import pytest

from llm_infer.response import BaseResolver, EventType, StreamEvent

pytestmark = pytest.mark.unit


class TestBaseResolverDispatch:
    """Test event dispatch to handlers."""

    def test_text_event_dispatch(self) -> None:
        """Test TEXT event is dispatched to on_text."""
        resolver = BaseResolver()
        event = StreamEvent(EventType.TEXT, "hello")
        resolver.handle(event)
        assert resolver._text_buffer == "hello"

    def test_think_start_dispatch(self) -> None:
        """Test THINK_START event is dispatched to on_think_start."""
        resolver = BaseResolver()
        resolver._think_buffer = "old content"
        event = StreamEvent(EventType.THINK_START)
        resolver.handle(event)
        assert resolver._think_buffer == ""

    def test_think_content_dispatch(self) -> None:
        """Test THINK_CONTENT event is dispatched to on_think_content."""
        resolver = BaseResolver()
        event = StreamEvent(EventType.THINK_CONTENT, "thinking")
        resolver.handle(event)
        assert resolver._think_buffer == "thinking"

    def test_code_start_dispatch(self) -> None:
        """Test CODE_START event captures language."""
        resolver = BaseResolver()
        event = StreamEvent(EventType.CODE_START, metadata={"language": "python"})
        resolver.handle(event)
        assert resolver._code_language == "python"
        assert resolver._code_buffer == ""

    def test_code_content_dispatch(self) -> None:
        """Test CODE_CONTENT event accumulates content."""
        resolver = BaseResolver()
        resolver.handle(StreamEvent(EventType.CODE_CONTENT, "line1"))
        resolver.handle(StreamEvent(EventType.CODE_CONTENT, "line2"))
        assert resolver._code_buffer == "line1line2"

    def test_unknown_event_type(self) -> None:
        """Test unknown event types don't raise errors."""
        resolver = BaseResolver()
        # This shouldn't raise even if we had an unknown type
        # (In practice all types are handled, but on_default exists for safety)
        event = StreamEvent(EventType.TEXT, "test")
        resolver.handle(event)  # Should not raise


class TestBaseResolverBuffers:
    """Test buffer management."""

    def test_initial_buffers_empty(self) -> None:
        """Test buffers are initially empty."""
        resolver = BaseResolver()
        assert resolver._text_buffer == ""
        assert resolver._think_buffer == ""
        assert resolver._code_buffer == ""
        assert resolver._code_language == ""

    def test_reset_clears_buffers(self) -> None:
        """Test reset clears all buffers and context stack."""
        resolver = BaseResolver()
        resolver._text_buffer = "text"
        resolver._think_buffer = "think"
        resolver._code_buffer = "code"
        resolver._code_language = "python"
        resolver._context_stack = [EventType.THINK_START]

        resolver.reset()

        assert resolver._text_buffer == ""
        assert resolver._think_buffer == ""
        assert resolver._code_buffer == ""
        assert resolver._code_language == ""
        assert resolver._context_stack == []


class TestBaseResolverSubclassing:
    """Test subclassing behavior."""

    def test_override_on_text(self) -> None:
        """Test overriding on_text handler."""
        collected: list[str] = []

        class CustomResolver(BaseResolver):
            def on_text(self, event: StreamEvent) -> None:
                collected.append(event.content)
                super().on_text(event)

        resolver = CustomResolver()
        resolver.handle(StreamEvent(EventType.TEXT, "hello"))
        resolver.handle(StreamEvent(EventType.TEXT, "world"))

        assert collected == ["hello", "world"]
        assert resolver._text_buffer == "helloworld"

    def test_override_on_code_end(self) -> None:
        """Test overriding on_code_end to process accumulated code."""
        executed: list[tuple[str, str]] = []

        class CodeExecutingResolver(BaseResolver):
            def on_code_end(self, event: StreamEvent, code: str, language: str) -> None:
                executed.append((language, code))
                super().on_code_end(event, code, language)

        resolver = CodeExecutingResolver()
        resolver.handle(
            StreamEvent(EventType.CODE_START, metadata={"language": "python"})
        )
        resolver.handle(StreamEvent(EventType.CODE_CONTENT, "print('hi')"))
        resolver.handle(StreamEvent(EventType.CODE_END))

        assert len(executed) == 1
        assert executed[0] == ("python", "print('hi')")

    def test_override_on_finish(self) -> None:
        """Test overriding on_finish hook for finalization."""
        finished = []

        class CustomResolver(BaseResolver):
            def on_finish(self) -> None:
                finished.append(True)
                super().on_finish()

        resolver = CustomResolver()
        resolver.finish()
        assert finished == [True]


class TestBaseResolverFullFlow:
    """Test complete event sequences."""

    def test_think_block_flow(self) -> None:
        """Test complete think block sequence."""
        resolver = BaseResolver()
        resolver.handle(StreamEvent(EventType.THINK_START))
        resolver.handle(StreamEvent(EventType.THINK_CONTENT, "part1"))
        resolver.handle(StreamEvent(EventType.THINK_CONTENT, "part2"))
        resolver.handle(StreamEvent(EventType.THINK_END))

        assert resolver._think_buffer == "part1part2"

    def test_code_block_flow(self) -> None:
        """Test complete code block sequence."""
        resolver = BaseResolver()
        resolver.handle(
            StreamEvent(EventType.CODE_START, metadata={"language": "rust"})
        )
        resolver.handle(StreamEvent(EventType.CODE_CONTENT, "fn main()"))
        resolver.handle(StreamEvent(EventType.CODE_CONTENT, " { }"))
        resolver.handle(StreamEvent(EventType.CODE_END))

        assert resolver._code_language == "rust"
        assert resolver._code_buffer == "fn main() { }"

    def test_mixed_content_flow(self) -> None:
        """Test mixed content sequence."""
        resolver = BaseResolver()
        resolver.handle(StreamEvent(EventType.TEXT, "before"))
        resolver.handle(StreamEvent(EventType.THINK_START))
        resolver.handle(StreamEvent(EventType.THINK_CONTENT, "thinking"))
        resolver.handle(StreamEvent(EventType.THINK_END))
        resolver.handle(StreamEvent(EventType.TEXT, "after"))

        assert resolver._text_buffer == "beforeafter"
        assert resolver._think_buffer == "thinking"


class TestBaseResolverContextTracking:
    """Test context tracking for nested structures."""

    def test_initial_context_empty(self) -> None:
        """Test context stack is initially empty."""
        resolver = BaseResolver()
        assert resolver._context_stack == []
        assert not resolver.in_think_context()
        assert not resolver.in_code_context()
        assert resolver.context_depth() == 0
        assert resolver.current_context() is None

    def test_think_context_tracking(self) -> None:
        """Test think block context is tracked."""
        resolver = BaseResolver()

        resolver.handle(StreamEvent(EventType.THINK_START))
        assert resolver.in_think_context()
        assert not resolver.in_code_context()
        assert resolver.context_depth() == 1
        assert resolver.current_context() == EventType.THINK_START

        resolver.handle(StreamEvent(EventType.THINK_CONTENT, "content"))
        assert resolver.in_think_context()

        resolver.handle(StreamEvent(EventType.THINK_END))
        assert not resolver.in_think_context()
        assert resolver.context_depth() == 0
        assert resolver.current_context() is None

    def test_code_context_tracking(self) -> None:
        """Test code block context is tracked."""
        resolver = BaseResolver()

        resolver.handle(StreamEvent(EventType.CODE_START, metadata={"language": "py"}))
        assert resolver.in_code_context()
        assert not resolver.in_think_context()
        assert resolver.context_depth() == 1
        assert resolver.current_context() == EventType.CODE_START

        resolver.handle(StreamEvent(EventType.CODE_CONTENT, "code"))
        assert resolver.in_code_context()

        resolver.handle(StreamEvent(EventType.CODE_END))
        assert not resolver.in_code_context()
        assert resolver.context_depth() == 0

    def test_nested_code_in_think_context(self) -> None:
        """Test nested code block inside think block."""
        resolver = BaseResolver()

        # Enter think block
        resolver.handle(StreamEvent(EventType.THINK_START))
        assert resolver.in_think_context()
        assert resolver.context_depth() == 1

        # Enter code block inside think
        resolver.handle(StreamEvent(EventType.CODE_START, metadata={"language": "py"}))
        assert resolver.in_think_context()  # Still in think
        assert resolver.in_code_context()  # Also in code
        assert resolver.context_depth() == 2
        assert resolver.current_context() == EventType.CODE_START  # Innermost

        # Exit code block
        resolver.handle(StreamEvent(EventType.CODE_END))
        assert resolver.in_think_context()
        assert not resolver.in_code_context()
        assert resolver.context_depth() == 1

        # Exit think block
        resolver.handle(StreamEvent(EventType.THINK_END))
        assert not resolver.in_think_context()
        assert resolver.context_depth() == 0

    def test_end_hook_receives_content(self) -> None:
        """Test that end hooks receive accumulated content."""
        received_think: list[str] = []
        received_code: list[tuple[str, str]] = []

        class TrackingResolver(BaseResolver):
            def on_think_end(self, event: StreamEvent, content: str) -> None:
                received_think.append(content)
                super().on_think_end(event, content)

            def on_code_end(self, event: StreamEvent, code: str, language: str) -> None:
                received_code.append((code, language))
                super().on_code_end(event, code, language)

        resolver = TrackingResolver()

        # Think block
        resolver.handle(StreamEvent(EventType.THINK_START))
        resolver.handle(StreamEvent(EventType.THINK_CONTENT, "thinking "))
        resolver.handle(StreamEvent(EventType.THINK_CONTENT, "content"))
        resolver.handle(StreamEvent(EventType.THINK_END))

        assert received_think == ["thinking content"]

        # Code block
        resolver.handle(
            StreamEvent(EventType.CODE_START, metadata={"language": "python"})
        )
        resolver.handle(StreamEvent(EventType.CODE_CONTENT, "print(1)"))
        resolver.handle(StreamEvent(EventType.CODE_END))

        assert received_code == [("print(1)", "python")]
