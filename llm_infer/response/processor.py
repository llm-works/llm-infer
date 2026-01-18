"""Response processor that wires parser and resolver together.

Provides a high-level interface for processing LLM response streams.
"""

from .parsers.think import ThinkTagParser
from .protocols import Parser, Resolver
from .resolvers.terminal import TerminalResolver


def create_default_parser() -> Parser:
    """Create the default parser (ThinkTagParser).

    Returns:
        A parser configured for typical use cases.
    """
    return ThinkTagParser()


class ResponseProcessor:
    """Wires parser and resolver together for stream processing.

    Provides a simple interface for processing LLM response streams.
    Uses sensible defaults but allows full customization.

    Example:
        # Basic usage with defaults
        processor = ResponseProcessor()
        for token in stream:
            processor.feed(token)
        processor.finish()

        # Custom resolver
        processor = ResponseProcessor(resolver=MyResolver())

        # Fully custom
        processor = ResponseProcessor(
            parser=MyParser(),
            resolver=MyResolver(),
        )
    """

    def __init__(
        self,
        parser: Parser | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        """Initialize the processor.

        Args:
            parser: Parser to use (default: ThinkTagParser).
            resolver: Resolver to use (default: TerminalResolver).
        """
        self._parser = parser or create_default_parser()
        self._resolver = resolver or TerminalResolver()

    @property
    def parser(self) -> Parser:
        """The parser being used."""
        return self._parser

    @property
    def resolver(self) -> Resolver:
        """The resolver being used."""
        return self._resolver

    def feed(self, token: str) -> None:
        """Process a token through the parser and resolver.

        Args:
            token: Text chunk from the stream.
        """
        for event in self._parser.feed(token):
            self._resolver.handle(event)

    def finish(self) -> None:
        """Finish processing and finalize output.

        Flushes any remaining content from the parser and calls
        the resolver's finish method.
        """
        for event in self._parser.flush():
            self._resolver.handle(event)
        self._resolver.finish()

    def reset(self) -> None:
        """Reset processor state for reuse.

        Resets both parser and resolver state.
        """
        self._parser.reset()
        self._resolver.reset()
