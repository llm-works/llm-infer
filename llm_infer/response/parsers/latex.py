"""LaTeX transformer for streaming responses.

Transforms LaTeX math notation to Unicode in text content.
"""

from collections.abc import Iterator

from ..events import EventType, StreamEvent
from ..latex import LatexConverter


class LatexTransformer:
    """Transforms LaTeX math notation to Unicode for streaming text.

    Takes raw string tokens and emits TEXT events with LaTeX commands
    converted to Unicode characters. Designed for use in a parser chain
    where it receives text from an upstream parser.

    Example:
        transformer = LatexTransformer()
        for token in stream:
            for event in transformer.feed(token):
                print(event.content)  # LaTeX converted to Unicode
    """

    def __init__(self) -> None:
        """Initialize the LaTeX transformer."""
        self._converter = LatexConverter()

    def feed(self, token: str) -> Iterator[StreamEvent]:
        """Process a token and yield events with LaTeX converted.

        Args:
            token: Text chunk from the stream.

        Yields:
            StreamEvent with LaTeX converted to Unicode.
        """
        converted = self._converter.process(token)
        if converted:
            yield StreamEvent(EventType.TEXT, converted)

    def flush(self) -> Iterator[StreamEvent]:
        """Flush remaining buffered content.

        Yields:
            StreamEvent for any remaining content.
        """
        remaining = self._converter.flush()
        if remaining:
            yield StreamEvent(EventType.TEXT, remaining)

    def reset(self) -> None:
        """Reset transformer state for reuse."""
        self._converter.reset()
