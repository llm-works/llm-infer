"""Protocol definitions for response processing.

Defines the abstract interfaces that parsers and resolvers must implement.
Consumers can implement these protocols to customize behavior.
"""

from collections.abc import Iterator
from typing import Protocol

from .events import StreamEvent


class Parser(Protocol):
    """Converts stream tokens to structured events.

    Parsers consume raw text tokens from LLM streams and emit structured
    StreamEvent objects. They handle buffering for incomplete patterns
    (e.g., partial tags split across chunks).

    Example:
        class MyParser:
            def feed(self, token: str) -> Iterator[StreamEvent]:
                # Parse token and yield events
                yield StreamEvent(EventType.TEXT, token)

            def flush(self) -> Iterator[StreamEvent]:
                # Emit any remaining buffered content
                return iter([])

            def reset(self) -> None:
                # Reset internal state
                pass
    """

    def feed(self, token: str) -> Iterator[StreamEvent]:
        """Process a token and yield any resulting events.

        Args:
            token: Text chunk from the stream (may be partial).

        Yields:
            StreamEvent objects for parsed content.
        """
        ...

    def flush(self) -> Iterator[StreamEvent]:
        """Flush any remaining buffered content.

        Call at end of stream to emit events for any buffered content
        that wasn't complete enough to emit during feed().

        Yields:
            StreamEvent objects for remaining content.
        """
        ...

    def reset(self) -> None:
        """Reset parser state for reuse.

        Clears internal buffers and state, allowing the parser to be
        reused for a new stream.
        """
        ...


class Resolver(Protocol):
    """Processes stream events into output.

    Resolvers receive structured events from parsers and produce output
    (render to terminal, execute code, extract data, etc.).

    Example:
        class MyResolver:
            def handle(self, event: StreamEvent) -> None:
                if event.type == EventType.TEXT:
                    print(event.content, end="")

            def finish(self) -> None:
                print()  # Final newline

            def reset(self) -> None:
                pass  # Reset any state
    """

    def handle(self, event: StreamEvent) -> None:
        """Process a single event.

        Args:
            event: The event to process.
        """
        ...

    def finish(self) -> None:
        """Called when the stream is complete.

        Perform any finalization (flush output, cleanup, etc.).
        """
        ...

    def reset(self) -> None:
        """Reset resolver state for reuse.

        Clears internal state, allowing the resolver to be reused
        for a new stream.
        """
        ...
