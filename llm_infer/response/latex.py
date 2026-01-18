"""LaTeX to Unicode converter for streaming text.

Converts LaTeX math notation to Unicode characters with streaming support.
"""

import re

# Simple command replacements
LATEX_REPLACEMENTS: dict[str, str] = {
    "\\times": "\u00d7",
    "\\div": "\u00f7",
    "\\pm": "\u00b1",
    "\\mp": "\u2213",
    "\\leq": "\u2264",
    "\\geq": "\u2265",
    "\\neq": "\u2260",
    "\\approx": "\u2248",
    "\\equiv": "\u2261",
    "\\infty": "\u221e",
    "\\pi": "\u03c0",
    "\\sum": "\u03a3",
    "\\prod": "\u03a0",
    "\\int": "\u222b",
    "\\partial": "\u2202",
    "\\nabla": "\u2207",
    "\\cdot": "\u00b7",
    "\\ldots": "\u2026",
    "\\cdots": "\u22ef",
    "\\to": "\u2192",
    "\\rightarrow": "\u2192",
    "\\leftarrow": "\u2190",
    "\\Rightarrow": "\u21d2",
    "\\Leftarrow": "\u21d0",
    "\\iff": "\u21d4",
    "\\forall": "\u2200",
    "\\exists": "\u2203",
    "\\in": "\u2208",
    "\\notin": "\u2209",
    "\\subset": "\u2282",
    "\\supset": "\u2283",
    "\\cup": "\u222a",
    "\\cap": "\u2229",
    "\\emptyset": "\u2205",
    "\\neg": "\u00ac",
    "\\land": "\u2227",
    "\\lor": "\u2228",
    "\\alpha": "\u03b1",
    "\\beta": "\u03b2",
    "\\gamma": "\u03b3",
    "\\delta": "\u03b4",
    "\\epsilon": "\u03b5",
    "\\zeta": "\u03b6",
    "\\eta": "\u03b7",
    "\\theta": "\u03b8",
    "\\lambda": "\u03bb",
    "\\mu": "\u03bc",
    "\\nu": "\u03bd",
    "\\xi": "\u03be",
    "\\rho": "\u03c1",
    "\\sigma": "\u03c3",
    "\\tau": "\u03c4",
    "\\phi": "\u03c6",
    "\\chi": "\u03c7",
    "\\psi": "\u03c8",
    "\\omega": "\u03c9",
    "\\Delta": "\u0394",
    "\\Gamma": "\u0393",
    "\\Theta": "\u0398",
    "\\Lambda": "\u039b",
    "\\Sigma": "\u03a3",
    "\\Phi": "\u03a6",
    "\\Psi": "\u03a8",
    "\\Omega": "\u03a9",
    "\\quad": " ",
    "\\qquad": "  ",
    "\\,": " ",
    "\\;": " ",
    "\\!": "",
}


class LatexConverter:
    """Converts LaTeX math notation to Unicode with streaming support.

    Handles streaming input by buffering potentially incomplete LaTeX commands.

    Example:
        converter = LatexConverter()
        for chunk in stream:
            print(converter.process(chunk), end="")
        print(converter.flush())
    """

    def __init__(self) -> None:
        """Initialize the converter."""
        self.buffer: str = ""

    def _apply_replacements(self, text: str) -> str:
        """Apply simple command replacements."""
        for latex, unicode_char in LATEX_REPLACEMENTS.items():
            text = text.replace(latex, unicode_char)
        return text

    def _process_frac(self, text: str) -> str:
        """Convert \\frac{a}{b} to a/b."""
        pattern = r"\\frac\{([^{}]*)\}\{([^{}]*)\}"
        return re.sub(pattern, r"\1/\2", text)

    def _process_sqrt(self, text: str) -> str:
        """Convert \\sqrt{x} to √x."""
        pattern = r"\\sqrt\{([^{}]*)\}"
        return re.sub(pattern, "\u221a\\1", text)

    def _process_boxed(self, text: str) -> str:
        """Convert \\boxed{x} to [x]."""
        pattern = r"\\boxed\{([^{}]*)\}"
        return re.sub(pattern, r"[\1]", text)

    def _process_text(self, text: str) -> str:
        """Convert \\text{...} to just the text."""
        pattern = r"\\text\{([^{}]*)\}"
        return re.sub(pattern, r"\1", text)

    def _strip_delimiters(self, text: str) -> str:
        """Strip $$ and $ math delimiters."""
        text = text.replace("$$", "")
        text = text.replace(" $ ", " ")
        text = text.replace("$ ", " ")
        text = text.replace(" $", " ")
        if text.startswith("$"):
            text = text[1:]
        if text.endswith("$"):
            text = text[:-1]
        return text

    def _convert(self, text: str) -> str:
        """Apply all LaTeX conversions."""
        text = self._strip_delimiters(text)
        text = self._process_frac(text)
        text = self._process_sqrt(text)
        text = self._process_boxed(text)
        text = self._process_text(text)
        text = self._apply_replacements(text)
        return text

    def _check_known_command(
        self, suffix: str, last_bs: int, text_len: int
    ) -> int | None:
        """Check if suffix matches a known command."""
        for cmd in sorted(LATEX_REPLACEMENTS.keys(), key=len, reverse=True):
            if suffix.startswith(cmd):
                after_cmd = suffix[len(cmd) :]
                return (
                    text_len if not after_cmd or not after_cmd[0].isalpha() else last_bs
                )
        return None

    def _check_brace_pattern(
        self, suffix: str, last_bs: int, text_len: int
    ) -> int | None:
        """Check for patterns like \\frac{."""
        for pattern in ["\\frac{", "\\sqrt{", "\\boxed{", "\\text{"]:
            if pattern.startswith(suffix) and len(suffix) < len(pattern):
                return last_bs
            if suffix.startswith(pattern):
                return last_bs if suffix.count("{") > suffix.count("}") else text_len
        return None

    def _find_safe_split(self, text: str) -> int:
        """Find position where we can safely split for processing."""
        last_bs = text.rfind("\\")
        if last_bs == -1:
            return len(text)

        suffix = text[last_bs:]
        if (pos := self._check_known_command(suffix, last_bs, len(text))) is not None:
            return pos
        if (pos := self._check_brace_pattern(suffix, last_bs, len(text))) is not None:
            return pos

        # Backslash + letters only = potentially incomplete command
        if len(suffix) <= 15 and suffix[1:].replace(" ", "").isalpha():
            return last_bs
        return len(text)

    def process(self, text: str) -> str:
        """Process text chunk, converting LaTeX to Unicode.

        Args:
            text: Text chunk (may be partial).

        Returns:
            Converted text. Incomplete commands are buffered.
        """
        self.buffer += text

        split_pos = self._find_safe_split(self.buffer)
        if split_pos == 0:
            return ""

        to_process = self.buffer[:split_pos]
        self.buffer = self.buffer[split_pos:]

        return self._convert(to_process)

    def flush(self) -> str:
        """Flush remaining buffer.

        Returns:
            Converted remaining content.
        """
        if not self.buffer:
            return ""
        result = self._convert(self.buffer)
        self.buffer = ""
        return result

    def reset(self) -> None:
        """Reset converter state for reuse."""
        self.buffer = ""
