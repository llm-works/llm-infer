"""Unit tests for TokenizerConfig."""

import pytest

from llm_infer.engines.native.tokenizer.config import TokenizerConfig

pytestmark = pytest.mark.unit


class TestTokenizerConfig:
    """Test TokenizerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = TokenizerConfig()
        assert config.pre_tokenizer_pattern is None

    def test_custom_pattern(self) -> None:
        """Test custom pre_tokenizer_pattern."""
        pattern = r"[a-zA-Z]+"
        config = TokenizerConfig(pre_tokenizer_pattern=pattern)
        assert config.pre_tokenizer_pattern == pattern

    def test_equality(self) -> None:
        """Test dataclass equality."""
        config1 = TokenizerConfig(pre_tokenizer_pattern="test")
        config2 = TokenizerConfig(pre_tokenizer_pattern="test")
        assert config1 == config2

    def test_inequality(self) -> None:
        """Test dataclass inequality."""
        config1 = TokenizerConfig(pre_tokenizer_pattern="test1")
        config2 = TokenizerConfig(pre_tokenizer_pattern="test2")
        assert config1 != config2
