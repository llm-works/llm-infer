"""LaTeX to Unicode converter for streaming output.

Converts LaTeX math notation to Unicode characters for terminal display
or other plain-text contexts.
"""

import re


class LatexFormatter:
    """Convert LaTeX math notation to Unicode for plain-text display.

    Handles streaming input by buffering incomplete commands that may be
    split across chunks. Supports common math symbols, Greek letters,
    and structured commands like \\frac, \\sqrt, \\boxed.

    Example:
        formatter = LatexFormatter()
        for chunk in stream:
            print(formatter.process(chunk), end="")
        print(formatter.flush())
    """

    # Simple command replacements (no arguments)
    REPLACEMENTS: dict[str, str] = {
        "\\times": "×",
        "\\div": "÷",
        "\\pm": "±",
        "\\mp": "∓",
        "\\leq": "≤",
        "\\geq": "≥",
        "\\neq": "≠",
        "\\approx": "≈",
        "\\equiv": "≡",
        "\\infty": "∞",
        "\\pi": "π",
        "\\sum": "Σ",
        "\\prod": "Π",
        "\\int": "∫",
        "\\partial": "∂",
        "\\nabla": "∇",
        "\\cdot": "·",
        "\\ldots": "…",
        "\\cdots": "⋯",
        "\\to": "→",
        "\\rightarrow": "→",
        "\\leftarrow": "←",
        "\\Rightarrow": "⇒",
        "\\Leftarrow": "⇐",
        "\\iff": "⇔",
        "\\forall": "∀",
        "\\exists": "∃",
        "\\in": "∈",
        "\\notin": "∉",
        "\\subset": "⊂",
        "\\supset": "⊃",
        "\\cup": "∪",
        "\\cap": "∩",
        "\\emptyset": "∅",
        "\\neg": "¬",
        "\\land": "∧",
        "\\lor": "∨",
        "\\alpha": "α",
        "\\beta": "β",
        "\\gamma": "γ",
        "\\delta": "δ",
        "\\epsilon": "ε",
        "\\zeta": "ζ",
        "\\eta": "η",
        "\\theta": "θ",
        "\\lambda": "λ",
        "\\mu": "μ",
        "\\nu": "ν",
        "\\xi": "ξ",
        "\\rho": "ρ",
        "\\sigma": "σ",
        "\\tau": "τ",
        "\\phi": "φ",
        "\\chi": "χ",
        "\\psi": "ψ",
        "\\omega": "ω",
        "\\Delta": "Δ",
        "\\Gamma": "Γ",
        "\\Theta": "Θ",
        "\\Lambda": "Λ",
        "\\Sigma": "Σ",
        "\\Phi": "Φ",
        "\\Psi": "Ψ",
        "\\Omega": "Ω",
        "\\quad": " ",
        "\\qquad": "  ",
        "\\,": " ",
        "\\;": " ",
        "\\!": "",
    }

    def __init__(self) -> None:
        self.buffer: str = ""

    def _apply_replacements(self, text: str) -> str:
        """Apply simple command replacements."""
        for latex, unicode_char in self.REPLACEMENTS.items():
            text = text.replace(latex, unicode_char)
        return text

    def _process_frac(self, text: str) -> str:
        """Convert \\frac{a}{b} to a/b."""
        pattern = r"\\frac\{([^{}]*)\}\{([^{}]*)\}"
        return re.sub(pattern, r"\1/\2", text)

    def _process_sqrt(self, text: str) -> str:
        """Convert \\sqrt{x} to √x."""
        pattern = r"\\sqrt\{([^{}]*)\}"
        return re.sub(pattern, r"√\1", text)

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

    def _check_known_command(
        self, suffix: str, last_bs: int, text_len: int
    ) -> int | None:
        """Check if suffix matches a known command. Returns split position or None."""
        for cmd in sorted(self.REPLACEMENTS.keys(), key=len, reverse=True):
            if suffix.startswith(cmd):
                after_cmd = suffix[len(cmd) :]
                return (
                    text_len if not after_cmd or not after_cmd[0].isalpha() else last_bs
                )
        return None

    def _check_brace_pattern(
        self, suffix: str, last_bs: int, text_len: int
    ) -> int | None:
        """Check for patterns like \\frac{. Returns split position or None."""
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

    def _convert(self, text: str) -> str:
        """Apply all LaTeX conversions."""
        text = self._strip_delimiters(text)
        text = self._process_frac(text)
        text = self._process_sqrt(text)
        text = self._process_boxed(text)
        text = self._process_text(text)
        text = self._apply_replacements(text)
        return text

    def process(self, text: str) -> str:
        """Process text chunk, converting LaTeX to Unicode.

        Buffers text to handle commands split across streaming chunks.

        Args:
            text: Text chunk to process (may be partial).

        Returns:
            Processed text with LaTeX converted to Unicode.
            Incomplete commands are buffered for the next call.
        """
        self.buffer += text

        split_pos = self._find_safe_split(self.buffer)

        if split_pos == 0:
            return ""  # Nothing safe to process yet

        to_process = self.buffer[:split_pos]
        self.buffer = self.buffer[split_pos:]

        return self._convert(to_process)

    def flush(self) -> str:
        """Flush remaining buffer at end of stream.

        Call this after all chunks have been processed to get any
        remaining buffered content.
        """
        if self.buffer:
            result = self._convert(self.buffer)
            self.buffer = ""
            return result
        return ""
