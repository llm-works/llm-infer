"""Base resolver implementation with hook methods.

Provides a base resolver that dispatches events to typed handler methods.
Subclasses can override specific handlers to customize behavior.
"""

from ..events import EventType, StreamEvent


class BaseResolver:
    """Base resolver with hook methods for each event type.

    Dispatches events to typed handler methods (on_text, on_think_start, etc.).
    Subclass and override specific handlers to customize behavior.

    Tracks nesting context via _context_stack, allowing subclasses to determine
    if they're inside a think block, code block, or nested structures.

    Example:
        class MyResolver(BaseResolver):
            def on_code_end(self, event: StreamEvent, code: str, language: str) -> None:
                if not self.in_think_context():
                    # Only execute code outside of think blocks
                    exec(code)
                super().on_code_end(event, code, language)

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
        self._context_stack: list[EventType] = []

    def handle(self, event: StreamEvent) -> None:
        """Process a single event by dispatching to typed handler.

        Args:
            event: The event to process.
        """
        handler_name = f"on_{event.type.name.lower()}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            self.on_default(event)
        elif event.type == EventType.THINK_END:
            handler(event, self._think_buffer)
        elif event.type == EventType.CODE_END:
            handler(event, self._code_buffer, self._code_language)
        else:
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
        self._context_stack.append(EventType.THINK_START)
        self._think_buffer = ""

    def on_think_content(self, event: StreamEvent) -> None:
        """Handle THINK_CONTENT event.

        Args:
            event: The think content event.
        """
        self._think_buffer += event.content

    def on_think_end(self, event: StreamEvent, content: str) -> None:
        """Handle THINK_END event.

        Args:
            event: The think end event.
            content: The accumulated think block content.
        """
        if self._context_stack and self._context_stack[-1] == EventType.THINK_START:
            self._context_stack.pop()

    def on_code_start(self, event: StreamEvent) -> None:
        """Handle CODE_START event.

        Args:
            event: The code start event.
        """
        self._context_stack.append(EventType.CODE_START)
        self._code_buffer = ""
        self._code_language = event.metadata.get("language", "")

    def on_code_content(self, event: StreamEvent) -> None:
        """Handle CODE_CONTENT event.

        Args:
            event: The code content event.
        """
        self._code_buffer += event.content

    def on_code_end(self, event: StreamEvent, code: str, language: str) -> None:
        """Handle CODE_END event.

        Args:
            event: The code end event.
            code: The accumulated code block content.
            language: The code block language (may be empty).
        """
        if self._context_stack and self._context_stack[-1] == EventType.CODE_START:
            self._context_stack.pop()

    def on_default(self, event: StreamEvent) -> None:
        """Handle unknown event types.

        Args:
            event: The unknown event.
        """
        pass  # Ignore unknown events by default

    def finish(self) -> None:
        """Called when the stream is complete.

        Calls on_finish() hook for subclass finalization.
        """
        self.on_finish()

    def on_finish(self) -> None:
        """Hook called when stream processing is complete.

        Override to perform finalization (flush output, cleanup, etc.).
        """
        pass

    def reset(self) -> None:
        """Reset resolver state for reuse."""
        self._text_buffer = ""
        self._think_buffer = ""
        self._code_buffer = ""
        self._code_language = ""
        self._context_stack = []

    # Context query methods

    def in_think_context(self) -> bool:
        """Check if currently inside a think block.

        Returns:
            True if a THINK_START has been seen without matching THINK_END.
        """
        return EventType.THINK_START in self._context_stack

    def in_code_context(self) -> bool:
        """Check if currently inside a code block.

        Returns:
            True if a CODE_START has been seen without matching CODE_END.
        """
        return EventType.CODE_START in self._context_stack

    def context_depth(self) -> int:
        """Get the current nesting depth.

        Returns:
            Number of unclosed block contexts.
        """
        return len(self._context_stack)

    def current_context(self) -> EventType | None:
        """Get the innermost context type.

        Returns:
            The most recent unclosed block type, or None if at top level.
        """
        return self._context_stack[-1] if self._context_stack else None
