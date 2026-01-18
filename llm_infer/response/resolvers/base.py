"""Base resolver implementation with hook methods.

Provides a base resolver that dispatches events to typed handler methods.
Subclasses can override specific handlers to customize behavior.
"""

from ..events import StreamEvent


class BaseResolver:
    """Base resolver with hook methods for each event type.

    Dispatches events to typed handler methods (on_text, on_think_start, etc.).
    Subclass and override specific handlers to customize behavior.

    Example:
        class MyResolver(BaseResolver):
            def on_code_end(self, event: StreamEvent) -> None:
                # Execute the accumulated code
                exec(self._code_buffer)
                super().on_code_end(event)

        resolver = MyResolver()
        for event in parser.feed(token):
            resolver.handle(event)
    """

    def __init__(self) -> None:
        """Initialize the resolver."""
        self._text_buffer: str = ""
        self._think_buffer: str = ""
        self._code_buffer: str = ""
        self._code_language: str = ""

    def handle(self, event: StreamEvent) -> None:
        """Process a single event by dispatching to typed handler.

        Args:
            event: The event to process.
        """
        handler_name = f"on_{event.type.name.lower()}"
        handler = getattr(self, handler_name, self.on_default)
        handler(event)

    def on_text(self, event: StreamEvent) -> None:
        """Handle TEXT event.

        Args:
            event: The text event.
        """
        self._text_buffer += event.content

    def on_think_start(self, event: StreamEvent) -> None:
        """Handle THINK_START event.

        Args:
            event: The think start event.
        """
        self._think_buffer = ""

    def on_think_content(self, event: StreamEvent) -> None:
        """Handle THINK_CONTENT event.

        Args:
            event: The think content event.
        """
        self._think_buffer += event.content

    def on_think_end(self, event: StreamEvent) -> None:
        """Handle THINK_END event.

        Args:
            event: The think end event.
        """
        pass  # Subclasses can process _think_buffer

    def on_code_start(self, event: StreamEvent) -> None:
        """Handle CODE_START event.

        Args:
            event: The code start event.
        """
        self._code_buffer = ""
        self._code_language = event.metadata.get("language", "")

    def on_code_content(self, event: StreamEvent) -> None:
        """Handle CODE_CONTENT event.

        Args:
            event: The code content event.
        """
        self._code_buffer += event.content

    def on_code_end(self, event: StreamEvent) -> None:
        """Handle CODE_END event.

        Args:
            event: The code end event.
        """
        pass  # Subclasses can process _code_buffer

    def on_default(self, event: StreamEvent) -> None:
        """Handle unknown event types.

        Args:
            event: The unknown event.
        """
        pass  # Ignore unknown events by default

    def finish(self) -> None:
        """Called when the stream is complete.

        Override to perform finalization (flush output, cleanup, etc.).
        """
        pass

    def reset(self) -> None:
        """Reset resolver state for reuse."""
        self._text_buffer = ""
        self._think_buffer = ""
        self._code_buffer = ""
        self._code_language = ""
