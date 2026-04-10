"""Unit tests for ThinkTagNormalizer, ThinkStreamSeparator, and extract_thinking.

The existing think_parser_test.py covers ThinkTagParser only. This file
covers the other classes/functions in the same module.
"""

from __future__ import annotations

import pytest

from llm_infer.response.parsers.think import (
    ThinkStreamSeparator,
    ThinkTagNormalizer,
    extract_thinking,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ThinkTagNormalizer
# ---------------------------------------------------------------------------


class TestThinkTagNormalizer:
    def test_no_variants_passthrough(self) -> None:
        """When there's only one tag (canonical), input is normalized + buffered."""
        n = ThinkTagNormalizer(["<think>"], ["</think>"])
        # The buffer keeps canonical-tag-length at the end for safety,
        # so process() output may be partial. flush() returns the rest.
        out = n.process("hello world body extra") + n.flush()
        assert out == "hello world body extra"

    def test_no_tags_at_all_passthrough(self) -> None:
        """Empty tag lists -> fast path returns text unchanged."""
        n = ThinkTagNormalizer([], [])
        assert n.process("hello") == "hello"
        assert n.flush() == ""

    def test_normalizes_variant_open_tag(self) -> None:
        n = ThinkTagNormalizer(["<think>", "<thinking>"], ["</think>", "</thinking>"])
        # Need enough text to flush past _variant_max_len buffer
        out = n.process("<thinking>hello</thinking> world extra")
        out += n.flush()
        assert "<think>" in out
        assert "</think>" in out
        assert "<thinking>" not in out
        assert "</thinking>" not in out

    def test_normalizes_variant_close_tag(self) -> None:
        n = ThinkTagNormalizer(["<think>", "<thinking>"], ["</think>", "</thinking>"])
        out = n.process("<think>hi</thinking> done with this") + n.flush()
        assert "</think>" in out
        assert "</thinking>" not in out

    def test_streaming_buffers_short_input(self) -> None:
        """Input shorter than variant_max_len is fully buffered."""
        n = ThinkTagNormalizer(["<think>", "<thinking>"], ["</think>", "</thinking>"])
        # "<th" is shorter than <thinking> -> buffered
        assert n.process("<th") == ""
        # Subsequent input completes the tag
        out = n.process("inking>hello world body more body</thinking>")
        out += n.flush()
        assert "<think>" in out
        assert "</think>" in out

    def test_flush_with_empty_buffer(self) -> None:
        n = ThinkTagNormalizer(["<think>", "<thinking>"], ["</think>", "</thinking>"])
        n.process("hello world this is plenty of text")
        n.flush()
        assert n.flush() == ""

    def test_flush_normalizes_remainder(self) -> None:
        n = ThinkTagNormalizer(["<think>", "<thinking>"], ["</think>", "</thinking>"])
        n.process("body content here is more")
        # Whatever's left in buffer at flush is normalized
        result = n.process("<thinking>") + n.flush()
        assert "<think>" in result
        assert "<thinking>" not in result


# ---------------------------------------------------------------------------
# ThinkStreamSeparator
# ---------------------------------------------------------------------------


class TestThinkStreamSeparator:
    def test_default_tags(self) -> None:
        s = ThinkStreamSeparator()
        # Process tokens with default tags
        thinking, content = s.process("<think>hidden</think>visible")
        # Need to flush remaining
        t2, c2 = s.flush()
        assert "hidden" in (thinking + t2)
        assert "visible" in (content + c2)

    def test_custom_tags(self) -> None:
        s = ThinkStreamSeparator(["<reason>"], ["</reason>"])
        thinking, content = s.process("<reason>why</reason>answer here is fine")
        t2, c2 = s.flush()
        assert "why" in (thinking + t2)
        assert "answer" in (content + c2)

    def test_no_think_block(self) -> None:
        s = ThinkStreamSeparator()
        thinking, content = s.process("just plain content here")
        t2, c2 = s.flush()
        assert (thinking + t2) == ""
        assert "just plain content here" in (content + c2)

    def test_streaming_split_across_tokens(self) -> None:
        s = ThinkStreamSeparator()
        all_thinking = ""
        all_content = ""
        for token in ["<thi", "nk>he", "llo</th", "ink>wor", "ld extra"]:
            t, c = s.process(token)
            all_thinking += t
            all_content += c
        t, c = s.flush()
        all_thinking += t
        all_content += c
        assert "hello" in all_thinking
        assert "world" in all_content

    def test_unclosed_think_block_flush(self) -> None:
        """An unclosed think block: text inside is emitted to thinking output."""
        s = ThinkStreamSeparator()
        all_thinking = ""
        for token in ["<think>", "incomplete reasoning that is long enough"]:
            t, _ = s.process(token)
            all_thinking += t
        t, c = s.flush()
        all_thinking += t
        assert "incomplete reasoning" in all_thinking
        assert c == ""

    def test_content_only_flush(self) -> None:
        s = ThinkStreamSeparator()
        s.process("ab")  # Below max_tag_len, buffered
        thinking, content = s.flush()
        assert thinking == ""
        assert content == "ab"

    def test_empty_flush(self) -> None:
        s = ThinkStreamSeparator()
        thinking, content = s.flush()
        assert thinking == ""
        assert content == ""

    def test_multiple_think_blocks(self) -> None:
        s = ThinkStreamSeparator()
        all_thinking = ""
        all_content = ""
        for token in [
            "before ",
            "<think>r1</think> middle <think>r2</think>",
            " after extra body",
        ]:
            t, c = s.process(token)
            all_thinking += t
            all_content += c
        t, c = s.flush()
        all_thinking += t
        all_content += c
        assert "r1" in all_thinking
        assert "r2" in all_thinking
        assert "before" in all_content
        assert "middle" in all_content
        assert "after" in all_content


# ---------------------------------------------------------------------------
# extract_thinking (non-streaming)
# ---------------------------------------------------------------------------


class TestExtractThinking:
    def test_no_think_block(self) -> None:
        thinking, content = extract_thinking("just plain content")
        assert thinking is None
        assert content == "just plain content"

    def test_single_think_block(self) -> None:
        thinking, content = extract_thinking("<think>reasoning</think>answer")
        assert thinking == "reasoning"
        assert content == "answer"

    def test_think_block_with_content_before_and_after(self) -> None:
        thinking, content = extract_thinking("intro <think>r</think> tail")
        assert thinking == "r"
        # Note: extract_thinking strips the result
        assert content == "intro  tail".strip()

    def test_multiple_think_blocks_concatenated(self) -> None:
        thinking, content = extract_thinking(
            "<think>r1</think>middle<think>r2</think>end"
        )
        # Multiple thinking parts joined
        assert thinking == "r1r2"
        assert "middle" in content
        assert "end" in content

    def test_unclosed_think_block(self) -> None:
        """Unclosed block: rest treated as thinking."""
        thinking, content = extract_thinking("<think>never closed")
        assert thinking == "never closed"
        assert content == ""

    def test_custom_tags(self) -> None:
        thinking, content = extract_thinking(
            "<reason>why</reason>answer",
            open_tags=["<reason>"],
            close_tags=["</reason>"],
        )
        assert thinking == "why"
        assert content == "answer"

    def test_thinking_variant_tag(self) -> None:
        thinking, content = extract_thinking("<thinking>r</thinking>answer")
        assert thinking == "r"
        assert content == "answer"

    def test_empty_text(self) -> None:
        thinking, content = extract_thinking("")
        assert thinking is None
        assert content == ""

    def test_whitespace_stripped(self) -> None:
        thinking, content = extract_thinking("  <think>r</think>  answer  ")
        assert thinking == "r"
        assert content == "answer"
