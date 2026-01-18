"""Base parser implementation with composition support.

Provides a composable parser that chains multiple parsers together.
"""

from collections.abc import Iterator

from ..events import StreamEvent
from ..protocols import Parser


class BaseParser:
    """Composable parser that chains multiple parsers.

    Events from one parser are fed as text content to the next parser,
    allowing layered parsing (e.g., think tags -> code blocks -> latex).

    Example:
        parser = BaseParser([ThinkTagParser(), CodeBlockParser()])
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
        """Recursively chain feed through parsers."""
        if len(parsers) == 1:
            yield from parsers[0].feed(token)
            return

        first, rest = parsers[0], parsers[1:]
        for event in first.feed(token):
            # Pass text content to next parser, emit other events directly
            if event.content:
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

        # Flush each parser and chain results through remaining parsers
        for i, parser in enumerate(self._parsers):
            for event in parser.flush():
                if i < len(self._parsers) - 1 and event.content:
                    # Chain through remaining parsers
                    yield from self._chain_feed(self._parsers[i + 1 :], event.content)
                else:
                    yield event

    def reset(self) -> None:
        """Reset all parsers in the chain."""
        for parser in self._parsers:
            parser.reset()
