"""Code block parser for streaming responses.

Parses markdown code fences and emits structured events.
"""

import re
from collections.abc import Iterator

from ..events import EventType, StreamEvent

# Length of closing fence pattern "```\n" - used for safe buffer splitting
_FENCE_CLOSE_LEN = 4


class CodeBlockParser:
    """Parser for markdown code fences.

    Detects code fences (```language ... ```) in streaming input and emits
    structured events. Handles fences split across chunks.

    Example:
        parser = CodeBlockParser()
        for token in stream:
            for event in parser.feed(token):
                print(event.type, event.content)
        for event in parser.flush():
            print(event.type, event.content)
    """

    # Regex to match opening fence with optional language
    _FENCE_OPEN_PATTERN = re.compile(r"^```(\w*)\n?")

    def __init__(self) -> None:
        """Initialize the code block parser."""
        self._in_code: bool = False
        self._language: str = ""
        self._buffer: str = ""

    def _try_match_open_fence(self) -> tuple[int, int, str] | None:
        """Try to match opening fence. Returns (start, end, language) or None."""
        # Look for ``` at any position
        idx = self._buffer.find("```")
        if idx == -1:
            return None

        # Try to match full fence pattern from that position
        match = self._FENCE_OPEN_PATTERN.match(self._buffer[idx:])
        if match:
            return idx, idx + match.end(), match.group(1)

        # Partial match - might be incomplete (e.g., "```py" without newline)
        # Check if there's more content after the backticks
        after_backticks = self._buffer[idx + 3 :]
        if after_backticks and "\n" not in after_backticks:
            # Could be incomplete language identifier
            return None

        return None

    def _find_safe_text_split(self) -> int:
        """Find position where we can safely split text for output.

        Returns the number of characters that are safe to emit, keeping any
        trailing backticks that could be part of a partial fence pattern.
        """
        # Count trailing backticks
        trailing_backticks = 0
        for i in range(len(self._buffer) - 1, -1, -1):
            if self._buffer[i] == "`":
                trailing_backticks += 1
            else:
                break

        if trailing_backticks == 0:
            # No backticks at end - safe to emit everything
            return len(self._buffer)

        if trailing_backticks >= 3:
            # We have 3+ backticks - could be incomplete fence with language
            # Keep all the backticks plus potential language chars
            backtick_start = len(self._buffer) - trailing_backticks
            # Check if there's content after the backticks (shouldn't be, since they're trailing)
            return backtick_start

        # 1-2 trailing backticks - keep them buffered
        return len(self._buffer) - trailing_backticks

    def _try_match_close_fence(self) -> int | None:
        """Try to match closing fence. Returns end position or None."""
        # Look for ``` on its own line (or at start)
        # Simple approach: find ``` that's either at start or after newline
        idx = 0
        while True:
            pos = self._buffer.find("```", idx)
            if pos == -1:
                return None
            # Check if it's at start or after newline
            if pos == 0 or self._buffer[pos - 1] == "\n":
                # Found closing fence - include trailing newline if present
                end = pos + 3
                if end < len(self._buffer) and self._buffer[end] == "\n":
                    end += 1
                return end
            idx = pos + 1

    def _process_in_code(self) -> Iterator[StreamEvent]:
        """Process buffer when inside code block."""
        close_pos = self._try_match_close_fence()
        if close_pos is None:
            # No close fence - emit safe content and buffer the rest
            # Keep enough buffer for potential fence pattern
            safe_len = max(0, len(self._buffer) - _FENCE_CLOSE_LEN)
            if safe_len > 0:
                yield StreamEvent(EventType.CODE_CONTENT, self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
            return

        # Found close fence
        # Content before the fence
        content_end = close_pos - 3  # Position of ```
        if self._buffer[content_end - 1 : content_end] == "\n":
            content_end -= 1  # Exclude newline before ```
        if content_end > 0:
            yield StreamEvent(EventType.CODE_CONTENT, self._buffer[:content_end])
        yield StreamEvent(EventType.CODE_END)
        self._buffer = self._buffer[close_pos:]
        self._in_code = False
        self._language = ""

    def _process_outside_code(self) -> Iterator[StreamEvent]:
        """Process buffer when outside code block."""
        result = self._try_match_open_fence()
        if result is None:
            # No open fence found - check for partial fence at end
            # We need to keep any trailing backticks that could become ```
            safe_len = self._find_safe_text_split()
            if safe_len > 0:
                yield StreamEvent(EventType.TEXT, self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
            return

        start_idx, end_idx, language = result
        # Found open fence
        if start_idx > 0:
            yield StreamEvent(EventType.TEXT, self._buffer[:start_idx])
        yield StreamEvent(EventType.CODE_START, metadata={"language": language})
        self._buffer = self._buffer[end_idx:]
        self._in_code = True
        self._language = language

    def feed(self, token: str) -> Iterator[StreamEvent]:
        """Process a token and yield events.

        Args:
            token: Text chunk from the stream (may be partial).

        Yields:
            StreamEvent objects for parsed content.
        """
        self._buffer += token

        while True:
            buffer_before = self._buffer
            if self._in_code:
                yield from self._process_in_code()
            else:
                yield from self._process_outside_code()

            # Break if no progress was made
            if self._buffer == buffer_before:
                break

    def flush(self) -> Iterator[StreamEvent]:
        """Flush remaining buffered content.

        Yields:
            StreamEvent objects for any remaining content.
        """
        if not self._buffer:
            return

        if self._in_code:
            yield StreamEvent(EventType.CODE_CONTENT, self._buffer)
            yield StreamEvent(EventType.CODE_END)
        else:
            yield StreamEvent(EventType.TEXT, self._buffer)
        self._buffer = ""

    def reset(self) -> None:
        """Reset parser state for reuse."""
        self._in_code = False
        self._language = ""
        self._buffer = ""
