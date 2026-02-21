"""Unit tests for ThinkTagNormalizer."""

import pytest

from llm_infer.response.parsers.think import ThinkTagNormalizer

# pytest convention: module-level marker applies to all tests in file
pytestmark = pytest.mark.unit


class TestThinkTagNormalizerBasic:
    """Test basic tag normalization."""

    def test_normalize_thinking_to_think(self) -> None:
        """Test that <thinking> is normalized to <think>."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = normalizer.process("<thinking>hello</thinking>") + normalizer.flush()
        assert result == "<think>hello</think>"

    def test_canonical_tags_unchanged(self) -> None:
        """Test that canonical tags pass through unchanged."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = normalizer.process("<think>hello</think>") + normalizer.flush()
        assert result == "<think>hello</think>"

    def test_no_tags_passthrough(self) -> None:
        """Test that text without tags passes through."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = normalizer.process("hello world") + normalizer.flush()
        assert result == "hello world"

    def test_multiple_tags(self) -> None:
        """Test multiple thinking blocks in one text."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = (
            normalizer.process("<thinking>one</thinking> text <thinking>two</thinking>")
            + normalizer.flush()
        )
        assert result == "<think>one</think> text <think>two</think>"

    def test_mixed_tags(self) -> None:
        """Test mix of canonical and variant tags."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = (
            normalizer.process("<think>one</think> and <thinking>two</thinking>")
            + normalizer.flush()
        )
        assert result == "<think>one</think> and <think>two</think>"


class TestThinkTagNormalizerStreaming:
    """Test streaming edge cases with partial tags."""

    def test_tag_split_across_chunks(self) -> None:
        """Test tag split across multiple chunks."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = ""
        result += normalizer.process("<thin")
        result += normalizer.process("king>hello</thi")
        result += normalizer.process("nking>")
        result += normalizer.flush()
        assert result == "<think>hello</think>"

    def test_single_char_chunks(self) -> None:
        """Test processing character by character."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        text = "<thinking>hi</thinking>"
        result = ""
        for char in text:
            result += normalizer.process(char)
        result += normalizer.flush()
        assert result == "<think>hi</think>"

    def test_partial_tag_at_end(self) -> None:
        """Test incomplete tag at end of stream."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = normalizer.process("hello<thin") + normalizer.flush()
        assert result == "hello<thin"

    def test_flush_empty_buffer(self) -> None:
        """Test flush with empty buffer."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        # Process and flush to empty the buffer
        _ = normalizer.process("short") + normalizer.flush()
        # Second flush should return empty string
        assert normalizer.flush() == ""

    def test_consecutive_flushes(self) -> None:
        """Test multiple flushes are safe."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        normalizer.process("hello")
        normalizer.flush()
        assert normalizer.flush() == ""
        assert normalizer.flush() == ""


class TestThinkTagNormalizerEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_input(self) -> None:
        """Test empty string input."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = normalizer.process("") + normalizer.flush()
        assert result == ""

    def test_empty_tags_list(self) -> None:
        """Test with empty tag lists - passthrough mode."""
        normalizer = ThinkTagNormalizer([], [])
        result = normalizer.process("<thinking>hello</thinking>") + normalizer.flush()
        # With empty lists, nothing to normalize, passes through
        assert result == "<thinking>hello</thinking>"

    def test_single_tag_list(self) -> None:
        """Test with single tag (nothing to normalize to)."""
        normalizer = ThinkTagNormalizer(["<think>"], ["</think>"])
        result = normalizer.process("<think>hello</think>") + normalizer.flush()
        assert result == "<think>hello</think>"

    def test_only_variant_tags_no_canonical(self) -> None:
        """Test when first tag is the only one, variants get normalized."""
        normalizer = ThinkTagNormalizer(
            ["<thought>", "<thinking>"], ["</thought>", "</thinking>"]
        )
        result = normalizer.process("<thinking>hello</thinking>") + normalizer.flush()
        assert result == "<thought>hello</thought>"

    def test_nested_angle_brackets(self) -> None:
        """Test content with angle brackets that aren't tags."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = (
            normalizer.process("<thinking>1 < 2 and 3 > 1</thinking>")
            + normalizer.flush()
        )
        assert result == "<think>1 < 2 and 3 > 1</think>"

    def test_unicode_content(self) -> None:
        """Test Unicode content inside tags."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = (
            normalizer.process("<thinking>こんにちは 🤔</thinking>")
            + normalizer.flush()
        )
        assert result == "<think>こんにちは 🤔</think>"

    def test_newlines_in_content(self) -> None:
        """Test multiline content inside tags."""
        normalizer = ThinkTagNormalizer(
            ["<think>", "<thinking>"], ["</think>", "</thinking>"]
        )
        result = (
            normalizer.process("<thinking>line1\nline2\nline3</thinking>")
            + normalizer.flush()
        )
        assert result == "<think>line1\nline2\nline3</think>"


class TestThinkTagNormalizerDefaults:
    """Test default canonical tags."""

    def test_default_canonical_with_empty_open(self) -> None:
        """Test default canonical tag when open_tags is empty."""
        normalizer = ThinkTagNormalizer([], ["</think>"])
        assert normalizer.canonical_open == "<think>"

    def test_default_canonical_with_empty_close(self) -> None:
        """Test default canonical tag when close_tags is empty."""
        normalizer = ThinkTagNormalizer(["<think>"], [])
        assert normalizer.canonical_close == "</think>"
