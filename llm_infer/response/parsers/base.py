"""Base parser implementation with composition support.

Provides a composable parser that chains multiple parsers together.
"""

from collections.abc import Iterator

from ..events import EventType, StreamEvent
from ..protocols import Parser


class BaseParser:
    """Composable parser that chains multiple parsers.

    Only TEXT events are passed through the parser chain. Other event types
    (THINK_*, CODE_*) are emitted directly without further processing.
    This allows layered text transformations while preserving semantic events.

    Example:
        parser = BaseParser([ThinkTagParser(), LatexTransformer()])
        for event in parser.feed(token):
            resolver.handle(event)
    """

    def __init__(self, parsers: list[Parser] | None = None) -> None:
        """Initialize with a list of parsers to chain.

        Args:
            parsers: Parsers to chain. Events flow through in order.
                     If empty or None, tokens pass through as TEXT events.
        """
        self._parsers: list[Parser] = parsers or []

    def feed(self, token: str) -> Iterator[StreamEvent]:
        """Process a token through the parser chain.

        Args:
            token: Text chunk from the stream.

        Yields:
            StreamEvent objects from the final parser in the chain.
        """
        if not self._parsers:
            from ..events import EventType

            yield StreamEvent(EventType.TEXT, token)
            return

        # Feed to first parser, then chain through remaining
        yield from self._chain_feed(self._parsers, token)

    def _chain_feed(self, parsers: list[Parser], token: str) -> Iterator[StreamEvent]:
        """Recursively chain feed through parsers.

        Only TEXT events with content are passed to downstream parsers.
        All other event types are emitted directly to preserve their semantics.
        """
        if len(parsers) == 1:
            yield from parsers[0].feed(token)
            return

        first, rest = parsers[0], parsers[1:]
        for event in first.feed(token):
            # Only chain TEXT events - other types preserve their semantics
            if event.type == EventType.TEXT and event.content:
                yield from self._chain_feed(rest, event.content)
            else:
                yield event

    def flush(self) -> Iterator[StreamEvent]:
        """Flush all parsers in the chain.

        Yields:
            StreamEvent objects from flushing all parsers.
        """
        if not self._parsers:
            return

        # Flush each parser and chain TEXT results through remaining parsers
        for i, parser in enumerate(self._parsers):
            for event in parser.flush():
                if (
                    i < len(self._parsers) - 1
                    and event.type == EventType.TEXT
                    and event.content
                ):
                    # Only chain TEXT events through remaining parsers
                    yield from self._chain_feed(self._parsers[i + 1 :], event.content)
                else:
                    yield event

    def reset(self) -> None:
        """Reset all parsers in the chain."""
        for parser in self._parsers:
            parser.reset()
