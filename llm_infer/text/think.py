"""Thinking block formatter for streaming output.

Formats <think>/<thinking> blocks with styling for terminal or other output.
"""

# ANSI escape codes for think block formatting
ANSI_THINK_START = "\x1b[3;38;5;245m"  # Italic + grey (256-color: 245)
ANSI_THINK_END = "\x1b[0m"  # Reset all formatting


class ThinkFormatter:
    """Formats thinking blocks with configurable styling for streaming output.

    Supports configurable tag variants used by different models (e.g., <think>,
    <thinking>). Default styling uses ANSI escape codes for terminal output
    (italic grey), but can be customized for web UIs or other contexts.

    Example:
        formatter = ThinkFormatter()
        for chunk in stream:
            print(formatter.process(chunk), end="")
        print(formatter.flush())
    """

    def __init__(
        self,
        open_tags: list[str] | None = None,
        close_tags: list[str] | None = None,
        style_start: str = ANSI_THINK_START,
        style_end: str = ANSI_THINK_END,
    ) -> None:
        """Initialize formatter with tag and style configuration.

        Args:
            open_tags: Opening tags to detect (default: ["<think>", "<thinking>"])
            close_tags: Closing tags to detect (default: ["</think>", "</thinking>"])
            style_start: String to insert before think content (default: ANSI italic grey)
            style_end: String to insert after think content (default: ANSI reset)
        """
        self.open_tags = open_tags or ["<think>", "<thinking>"]
        self.close_tags = close_tags or ["</think>", "</thinking>"]
        self.style_start = style_start
        self.style_end = style_end
        self.max_tag_len = max(
            max((len(t) for t in self.open_tags), default=0),
            max((len(t) for t in self.close_tags), default=0),
        )
        self.in_think: bool = False
        self.buffer: str = ""

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
        """Process buffer when inside think block. Returns (output, should_break)."""
        end_idx, tag_len = self._find_close_tag(self.buffer)
        if end_idx == -1:
            safe_len = max(0, len(self.buffer) - self.max_tag_len)
            output = (
                self.style_start + self.buffer[:safe_len] + self.style_end
                if safe_len > 0
                else ""
            )
            self.buffer = self.buffer[safe_len:]
            return output, True
        output = self.style_start + self.buffer[:end_idx] + self.style_end
        self.buffer, self.in_think = self.buffer[end_idx + tag_len :], False
        return output, False

    def _process_outside_think(self) -> tuple[str, bool]:
        """Process buffer when outside think block. Returns (output, should_break)."""
        start_idx, tag_len = self._find_open_tag(self.buffer)
        if start_idx == -1:
            safe_len = max(0, len(self.buffer) - self.max_tag_len)
            output = self.buffer[:safe_len]
            self.buffer = self.buffer[safe_len:]
            return output, True
        output = self.buffer[:start_idx]
        self.buffer, self.in_think = self.buffer[start_idx + tag_len :], True
        return output, False

    def process(self, text: str) -> str:
        """Process text chunk, applying formatting to think blocks.

        Args:
            text: Text chunk to process (may be partial).

        Returns:
            Processed text with styling applied to complete think blocks.
            Incomplete tags are buffered for the next call.
        """
        self.buffer += text
        output = ""
        while True:
            chunk, should_break = (
                self._process_in_think()
                if self.in_think
                else self._process_outside_think()
            )
            output += chunk
            if should_break:
                break
        return output

    def flush(self) -> str:
        """Flush remaining buffer at end of stream.

        Call this after all chunks have been processed to get any
        remaining buffered content. Clears the buffer after returning.
        """
        if self.buffer:
            output = self.buffer
            self.buffer = ""
            if self.in_think:
                return self.style_start + output + self.style_end
            return output
        return ""
