"""Terminal resolver for ANSI-styled output.

Renders stream events to terminal with ANSI styling for think blocks.
"""

import sys
from typing import TextIO

from ..events import StreamEvent
from ..latex import LatexConverter
from .base import BaseResolver

# ANSI escape codes for styling
ANSI_THINK_START = "\x1b[3;38;5;245m"  # Italic + grey (256-color: 245)
ANSI_THINK_END = "\x1b[0m"  # Reset all formatting


class TerminalResolver(BaseResolver):
    """Resolver that outputs to terminal with ANSI styling.

    Renders text directly, think blocks in italic grey, and code blocks
    with their content. Provides the same output behavior as the current
    QueryTool streaming implementation.

    Example:
        resolver = TerminalResolver()
        for event in parser.feed(token):
            resolver.handle(event)
        resolver.finish()

        # With LaTeX conversion
        resolver = TerminalResolver(convert_latex=True)
    """

    def __init__(
        self,
        output: TextIO | None = None,
        think_style_start: str = ANSI_THINK_START,
        think_style_end: str = ANSI_THINK_END,
        convert_latex: bool = False,
    ) -> None:
        """Initialize the terminal resolver.

        Args:
            output: Output stream (default: sys.stdout).
            think_style_start: ANSI sequence for think block start.
            think_style_end: ANSI sequence for think block end.
            convert_latex: If True, convert LaTeX math notation to Unicode.
        """
        super().__init__()
        self._output = output or sys.stdout
        self._think_style_start = think_style_start
        self._think_style_end = think_style_end
        self._convert_latex = convert_latex
        self._latex_formatter: LatexConverter | None = None
        if convert_latex:
            self._latex_formatter = LatexConverter()

    def _write(self, text: str, apply_latex: bool = True) -> None:
        """Write text to output and flush.

        Args:
            text: Text to write.
            apply_latex: If True and LaTeX conversion is enabled, convert text.
        """
        if apply_latex and self._latex_formatter:
            text = self._latex_formatter.process(text)
        if text:
            self._output.write(text)
            self._output.flush()

    def on_text(self, event: StreamEvent) -> None:
        """Handle TEXT event - write directly to output."""
        if event.content:
            self._write(event.content)

    def on_think_start(self, event: StreamEvent) -> None:
        """Handle THINK_START event - start italic grey styling."""
        super().on_think_start(event)
        self._write(self._think_style_start, apply_latex=False)

    def on_think_content(self, event: StreamEvent) -> None:
        """Handle THINK_CONTENT event - write styled content."""
        super().on_think_content(event)
        if event.content:
            self._write(event.content)

    def on_think_end(self, event: StreamEvent, content: str) -> None:
        """Handle THINK_END event - reset styling."""
        super().on_think_end(event, content)
        self._write(self._think_style_end, apply_latex=False)

    def on_code_start(self, event: StreamEvent) -> None:
        """Handle CODE_START event - write fence."""
        super().on_code_start(event)
        language = event.metadata.get("language", "")
        self._write(f"```{language}\n", apply_latex=False)

    def on_code_content(self, event: StreamEvent) -> None:
        """Handle CODE_CONTENT event - write code content."""
        super().on_code_content(event)
        if event.content:
            # Don't convert LaTeX in code blocks
            self._write(event.content, apply_latex=False)

    def on_code_end(self, event: StreamEvent, code: str, language: str) -> None:
        """Handle CODE_END event - write closing fence."""
        super().on_code_end(event, code, language)
        self._write("\n```\n", apply_latex=False)

    def on_finish(self) -> None:
        """Finalize output."""
        # Flush LaTeX formatter if present
        if self._latex_formatter:
            remaining = self._latex_formatter.flush()
            if remaining:
                self._output.write(remaining)
                self._output.flush()

        # Ensure styling is reset if stream ended mid-think
        if self.in_think_context():
            self._write(self._think_style_end, apply_latex=False)

        # Ensure code block is closed if stream ended mid-code
        if self.in_code_context():
            self._write("\n```\n", apply_latex=False)

    def reset(self) -> None:
        """Reset resolver state."""
        super().reset()
        if self._latex_formatter:
            self._latex_formatter.reset()
