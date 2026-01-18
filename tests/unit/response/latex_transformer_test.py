"""Unit tests for LatexTransformer."""

import pytest

from llm_infer.response import LatexTransformer, StreamEvent

pytestmark = pytest.mark.unit


def collect_events(transformer: LatexTransformer, text: str) -> list[StreamEvent]:
    """Helper to collect all events from feeding text and flushing."""
    events = list(transformer.feed(text))
    events.extend(transformer.flush())
    return events


def get_content(events: list[StreamEvent]) -> str:
    """Helper to extract text content from events."""
    return "".join(e.content for e in events)


class TestLatexTransformerBasic:
    """Test basic LaTeX transformation."""

    def test_plain_text(self) -> None:
        """Test plain text without LaTeX."""
        transformer = LatexTransformer()
        events = collect_events(transformer, "hello world")
        content = get_content(events)
        assert content == "hello world"

    def test_simple_symbol(self) -> None:
        """Test simple symbol replacement."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"x \times y")
        content = get_content(events)
        assert "\u00d7" in content  # multiplication sign

    def test_greek_letters(self) -> None:
        """Test Greek letter replacements."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"\alpha + \beta = \gamma")
        content = get_content(events)
        assert "\u03b1" in content  # alpha
        assert "\u03b2" in content  # beta
        assert "\u03b3" in content  # gamma

    def test_math_operators(self) -> None:
        """Test math operator replacements."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"a \leq b \geq c")
        content = get_content(events)
        assert "\u2264" in content  # <=
        assert "\u2265" in content  # >=

    def test_arrows(self) -> None:
        """Test arrow replacements."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"a \to b \Rightarrow c")
        content = get_content(events)
        assert "\u2192" in content  # right arrow
        assert "\u21d2" in content  # double right arrow


class TestLatexTransformerStructured:
    """Test structured LaTeX commands."""

    def test_frac(self) -> None:
        """Test fraction conversion."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"\frac{a}{b}")
        content = get_content(events)
        assert content == "a/b"

    def test_sqrt(self) -> None:
        """Test square root conversion."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"\sqrt{x}")
        content = get_content(events)
        assert "\u221ax" == content

    def test_boxed(self) -> None:
        """Test boxed conversion."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"\boxed{answer}")
        content = get_content(events)
        assert content == "[answer]"

    def test_text(self) -> None:
        """Test text command removal."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"\text{hello}")
        content = get_content(events)
        assert content == "hello"


class TestLatexTransformerDelimiters:
    """Test math delimiter handling."""

    def test_dollar_signs(self) -> None:
        """Test $ delimiter stripping."""
        transformer = LatexTransformer()
        events = collect_events(transformer, "$x + y$")
        content = get_content(events)
        assert "$" not in content
        assert "x + y" in content

    def test_double_dollar_signs(self) -> None:
        """Test $$ delimiter stripping."""
        transformer = LatexTransformer()
        events = collect_events(transformer, "$$x + y$$")
        content = get_content(events)
        assert "$$" not in content


class TestLatexTransformerStreaming:
    """Test streaming edge cases."""

    def test_command_split_across_chunks(self) -> None:
        """Test command split across chunks."""
        transformer = LatexTransformer()
        events = []
        events.extend(transformer.feed(r"\al"))
        events.extend(transformer.feed(r"pha"))
        events.extend(transformer.flush())
        content = get_content(events)
        assert "\u03b1" in content

    def test_brace_pattern_split(self) -> None:
        """Test brace pattern split across chunks."""
        transformer = LatexTransformer()
        events = []
        events.extend(transformer.feed(r"\frac{a"))
        events.extend(transformer.feed(r"}{b}"))
        events.extend(transformer.flush())
        content = get_content(events)
        assert "a/b" in content

    def test_incomplete_command_at_end(self) -> None:
        """Test incomplete command at end of stream."""
        transformer = LatexTransformer()
        events = collect_events(transformer, r"hello \alp")
        content = get_content(events)
        # Should output incomplete command as-is
        assert "hello" in content


class TestLatexTransformerReset:
    """Test transformer reset functionality."""

    def test_reset_clears_state(self) -> None:
        """Test that reset clears internal state."""
        transformer = LatexTransformer()
        # Feed partial command
        list(transformer.feed(r"\alp"))
        # Reset
        transformer.reset()
        # Should behave as fresh transformer
        events = collect_events(transformer, "fresh text")
        content = get_content(events)
        assert content == "fresh text"
