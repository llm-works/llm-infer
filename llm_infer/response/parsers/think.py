"""Think tag parser and utilities for streaming responses.

Parses <think>/<thinking> blocks and emits structured events.
Also provides utilities for tag normalization and content extraction.
"""

from collections.abc import Iterator

from ..events import EventType, StreamEvent


class ThinkTagParser:
    """Parser for think/thinking tag blocks.

    Detects opening and closing think tags in streaming input and emits
    structured events. Handles tag variants (e.g., <think>, <thinking>)
    and buffering for tags split across chunks.

    Example:
        parser = ThinkTagParser()
        for token in stream:
            for event in parser.feed(token):
                print(event.type, event.content)
        for event in parser.flush():
            print(event.type, event.content)
    """

    def __init__(
        self,
        open_tags: list[str] | None = None,
        close_tags: list[str] | None = None,
    ) -> None:
        """Initialize parser with tag configuration.

        Args:
            open_tags: Opening tags to detect (default: ["<think>", "<thinking>"]).
            close_tags: Closing tags to detect (default: ["</think>", "</thinking>"]).
        """
        self.open_tags = open_tags or ["<think>", "<thinking>"]
        self.close_tags = close_tags or ["</think>", "</thinking>"]
        self._max_tag_len = max(
            max((len(t) for t in self.open_tags), default=0),
            max((len(t) for t in self.close_tags), default=0),
        )
        self._in_think: bool = False
        self._buffer: str = ""

    def _find_open_tag(self, text: str) -> tuple[int, int]:
        """Find earliest opening tag. Returns (position, tag_length) or (-1, 0)."""
        best_pos, best_len = -1, 0
        for tag in self.open_tags:
            pos = text.find(tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best_len = pos, len(tag)
        return best_pos, best_len

    def _find_close_tag(self, text: str) -> tuple[int, int]:
        """Find earliest closing tag. Returns (position, tag_length) or (-1, 0)."""
        best_pos, best_len = -1, 0
        for tag in self.close_tags:
            pos = text.find(tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best_len = pos, len(tag)
        return best_pos, best_len

    def _process_in_think(self) -> Iterator[StreamEvent]:
        """Process buffer when inside think block."""
        end_idx, tag_len = self._find_close_tag(self._buffer)
        if end_idx == -1:
            # No close tag found - emit safe content and buffer the rest
            safe_len = max(0, len(self._buffer) - self._max_tag_len)
            if safe_len > 0:
                yield StreamEvent(EventType.THINK_CONTENT, self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
            return

        # Found close tag
        if end_idx > 0:
            yield StreamEvent(EventType.THINK_CONTENT, self._buffer[:end_idx])
        yield StreamEvent(EventType.THINK_END)
        self._buffer = self._buffer[end_idx + tag_len :]
        self._in_think = False

    def _process_outside_think(self) -> Iterator[StreamEvent]:
        """Process buffer when outside think block."""
        start_idx, tag_len = self._find_open_tag(self._buffer)
        if start_idx == -1:
            # No open tag found - emit safe content and buffer the rest
            safe_len = max(0, len(self._buffer) - self._max_tag_len)
            if safe_len > 0:
                yield StreamEvent(EventType.TEXT, self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
            return

        # Found open tag
        if start_idx > 0:
            yield StreamEvent(EventType.TEXT, self._buffer[:start_idx])
        yield StreamEvent(EventType.THINK_START)
        self._buffer = self._buffer[start_idx + tag_len :]
        self._in_think = True

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
            if self._in_think:
                yield from self._process_in_think()
            else:
                yield from self._process_outside_think()

            # Break if no progress was made (waiting for more input)
            if self._buffer == buffer_before:
                break

    def flush(self) -> Iterator[StreamEvent]:
        """Flush remaining buffered content.

        Yields:
            StreamEvent objects for any remaining content.
        """
        if not self._buffer:
            return

        # Process any complete tags remaining in buffer before final flush
        while self._buffer:
            buffer_before = self._buffer
            if self._in_think:
                yield from self._process_in_think()
            else:
                yield from self._process_outside_think()
            # If no progress, emit remaining buffer and break
            if self._buffer == buffer_before:
                break

        # Emit any remaining content that couldn't be processed
        if self._buffer:
            if self._in_think:
                yield StreamEvent(EventType.THINK_CONTENT, self._buffer)
                yield StreamEvent(EventType.THINK_END)
            else:
                yield StreamEvent(EventType.TEXT, self._buffer)
            self._buffer = ""

    def reset(self) -> None:
        """Reset parser state for reuse."""
        self._in_think = False
        self._buffer = ""


class ThinkTagNormalizer:
    """Normalizes think tags to canonical format for streaming output.

    Replaces any variant tags (e.g., <thinking>) with the canonical tag
    (first in the list). Handles streaming by buffering incomplete tags.

    Example:
        # Config: tags_open=["<think>", "<thinking>"], tags_close=["</think>", "</thinking>"]
        normalizer = ThinkTagNormalizer(["<think>", "<thinking>"], ["</think>", "</thinking>"])
        # Input: "<thinking>hello</thinking>" -> Output: "<think>hello</think>"
    """

    def __init__(self, open_tags: list[str], close_tags: list[str]) -> None:
        """Initialize normalizer with tags from model config.

        The first tag in each list is canonical; others are normalized to it.

        Args:
            open_tags: Opening tags from model config (first is canonical).
            close_tags: Closing tags from model config (first is canonical).
        """
        self.open_tags = open_tags
        self.close_tags = close_tags
        self.canonical_open = open_tags[0] if open_tags else "<think>"
        self.canonical_close = close_tags[0] if close_tags else "</think>"
        # Buffer size based on longest variant tag (to catch complete tags before split)
        self._variant_max_len = max(
            max((len(t) for t in self.open_tags), default=0),
            max((len(t) for t in self.close_tags), default=0),
        )
        # After normalization, buffer based on canonical tag length
        self._canonical_max_len = max(
            len(self.canonical_open), len(self.canonical_close)
        )
        self._buffer = ""

    def _normalize(self, text: str) -> str:
        """Replace variant tags with canonical tags."""
        for tag in self.open_tags[1:]:  # Skip first (canonical)
            text = text.replace(tag, self.canonical_open)
        for tag in self.close_tags[1:]:  # Skip first (canonical)
            text = text.replace(tag, self.canonical_close)
        return text

    def process(self, text: str) -> str:
        """Process text chunk, normalizing think tags.

        Args:
            text: Text chunk to process (may be partial).

        Returns:
            Text with variant tags replaced by canonical tags.
            Incomplete tags are buffered for the next call.
        """
        # Fast path: no tags to normalize
        if self._variant_max_len == 0:
            return text

        self._buffer += text

        # Buffer enough to catch longest variant tag, then normalize
        if len(self._buffer) <= self._variant_max_len:
            return ""

        # Normalize complete tags in buffer before splitting
        self._buffer = self._normalize(self._buffer)

        # After normalization, keep canonical tag length at end for partial tag safety
        safe_len = max(0, len(self._buffer) - self._canonical_max_len)
        if safe_len == 0:
            return ""

        output = self._buffer[:safe_len]
        self._buffer = self._buffer[safe_len:]
        return output

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        if not self._buffer:
            return ""
        output = self._buffer
        self._buffer = ""
        return self._normalize(output)


class ThinkStreamSeparator:
    """Streaming separator that routes tokens to thinking or content fields.

    For use in streaming responses where we need to emit tokens to
    either `thinking` or `content` fields based on whether we're inside
    a think block.

    Example:
        separator = ThinkStreamSeparator()
        for token in stream:
            thinking, content = separator.process(token)
            if thinking:
                emit_to_thinking_field(thinking)
            if content:
                emit_to_content_field(content)
        thinking, content = separator.flush()
    """

    def __init__(
        self,
        open_tags: list[str] | None = None,
        close_tags: list[str] | None = None,
    ) -> None:
        """Initialize separator with tag configuration.

        Args:
            open_tags: Opening tags to detect (default: ["<think>", "<thinking>"]).
            close_tags: Closing tags to detect (default: ["</think>", "</thinking>"]).
        """
        self.open_tags = open_tags or ["<think>", "<thinking>"]
        self.close_tags = close_tags or ["</think>", "</thinking>"]
        self._max_tag_len = max(
            max((len(t) for t in self.open_tags), default=0),
            max((len(t) for t in self.close_tags), default=0),
        )
        self._in_think: bool = False
        self._buffer: str = ""

    def _find_open_tag(self, text: str) -> tuple[int, int]:
        """Find earliest opening tag. Returns (position, tag_length) or (-1, 0)."""
        best_pos, best_len = -1, 0
        for tag in self.open_tags:
            pos = text.find(tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best_len = pos, len(tag)
        return best_pos, best_len

    def _find_close_tag(self, text: str) -> tuple[int, int]:
        """Find earliest closing tag. Returns (position, tag_length) or (-1, 0)."""
        best_pos, best_len = -1, 0
        for tag in self.close_tags:
            pos = text.find(tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best_len = pos, len(tag)
        return best_pos, best_len

    def _process_in_think(self) -> tuple[str, bool]:
        """Process buffer inside think block. Returns (thinking_text, should_break)."""
        end_idx, tag_len = self._find_close_tag(self._buffer)
        if end_idx == -1:
            safe_len = max(0, len(self._buffer) - self._max_tag_len)
            if safe_len > 0:
                text = self._buffer[:safe_len]
                self._buffer = self._buffer[safe_len:]
                return text, True
            return "", True
        text = self._buffer[:end_idx]
        self._buffer = self._buffer[end_idx + tag_len :]
        self._in_think = False
        return text, False

    def _process_outside_think(self) -> tuple[str, bool]:
        """Process buffer outside think block. Returns (content_text, should_break)."""
        start_idx, tag_len = self._find_open_tag(self._buffer)
        if start_idx == -1:
            safe_len = max(0, len(self._buffer) - self._max_tag_len)
            if safe_len > 0:
                text = self._buffer[:safe_len]
                self._buffer = self._buffer[safe_len:]
                return text, True
            return "", True
        text = self._buffer[:start_idx]
        self._buffer = self._buffer[start_idx + tag_len :]
        self._in_think = True
        return text, False

    def process(self, token: str) -> tuple[str, str]:
        """Process a token and return (thinking, content) to emit."""
        self._buffer += token
        thinking_out, content_out = "", ""

        while True:
            if self._in_think:
                text, should_break = self._process_in_think()
                thinking_out += text
            else:
                text, should_break = self._process_outside_think()
                content_out += text
            if should_break:
                break

        return thinking_out, content_out

    def flush(self) -> tuple[str, str]:
        """Flush remaining buffer.

        Returns:
            Tuple of (thinking, content) for remaining buffered content.
        """
        if not self._buffer:
            return "", ""

        if self._in_think:
            result = (self._buffer, "")
        else:
            result = ("", self._buffer)
        self._buffer = ""
        return result


def _find_earliest_tag(text: str, tags: list[str], start: int = 0) -> tuple[int, int]:
    """Find earliest matching tag in text. Returns (position, tag_length) or (-1, 0)."""
    best_pos, best_len = -1, 0
    for tag in tags:
        tag_pos = text.find(tag, start)
        if tag_pos != -1 and (best_pos == -1 or tag_pos < best_pos):
            best_pos, best_len = tag_pos, len(tag)
    return best_pos, best_len


def _extract_think_block(
    text: str, pos: int, open_tags: list[str], close_tags: list[str]
) -> tuple[str | None, str, int] | None:
    """Extract a single think block. Returns (thinking, content_before, new_pos) or None."""
    open_pos, open_len = _find_earliest_tag(text, open_tags, pos)
    if open_pos == -1:
        return None

    content_before = text[pos:open_pos] if open_pos > pos else ""
    think_start = open_pos + open_len

    close_pos, close_len = _find_earliest_tag(text, close_tags, think_start)
    if close_pos == -1:
        # Unclosed block - treat rest as thinking
        return text[think_start:], content_before, len(text)

    return text[think_start:close_pos], content_before, close_pos + close_len


def extract_thinking(
    text: str,
    open_tags: list[str] | None = None,
    close_tags: list[str] | None = None,
) -> tuple[str | None, str]:
    """Extract thinking content from text, separating it from main content.

    Args:
        text: Text potentially containing think blocks.
        open_tags: Opening tags to detect (default: ["<think>", "<thinking>"]).
        close_tags: Closing tags to detect (default: ["</think>", "</thinking>"]).

    Returns:
        Tuple of (thinking_content, main_content):
        - thinking_content: Combined content from all think blocks, or None if no blocks.
        - main_content: Text with think blocks removed.
    """
    open_tags = open_tags or ["<think>", "<thinking>"]
    close_tags = close_tags or ["</think>", "</thinking>"]

    thinking_parts: list[str] = []
    content_parts: list[str] = []
    pos = 0

    while pos < len(text):
        result = _extract_think_block(text, pos, open_tags, close_tags)
        if result is None:
            content_parts.append(text[pos:])
            break

        thinking, content_before, pos = result
        if content_before:
            content_parts.append(content_before)
        if thinking:
            thinking_parts.append(thinking)

    thinking = "".join(thinking_parts).strip() if thinking_parts else None
    content = "".join(content_parts).strip()
    return thinking, content
