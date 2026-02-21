"""Unit tests for EngineConfig."""

import tempfile

import pytest
import torch

from llm_infer.engines.native.config import EngineConfig
from llm_infer.engines.native.model.config import TransformerConfig

pytestmark = pytest.mark.unit


def make_model_config() -> TransformerConfig:
    """Create a test TransformerConfig."""
    return TransformerConfig(
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        intermediate_size=14336,
        vocab_size=32000,
    )


class TestEngineConfigProperties:
    """Test EngineConfig properties."""

    def test_max_seq_len(self) -> None:
        """Test max_seq_len calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EngineConfig(
                model=make_model_config(),
                model_path=tmpdir,
                num_blocks=100,
                block_size=16,
            )
            assert config.max_seq_len == 1600  # 100 * 16


class TestEngineConfigValidation:
    """Test EngineConfig.validate method."""

    def test_validate_positive_block_size(self) -> None:
        """Test that block_size must be positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EngineConfig(
                model=make_model_config(),
                model_path=tmpdir,
                block_size=0,
            )
            with pytest.raises(ValueError, match="block_size"):
                config.validate()

    def test_validate_positive_num_blocks(self) -> None:
        """Test that num_blocks must be positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EngineConfig(
                model=make_model_config(),
                model_path=tmpdir,
                num_blocks=0,
            )
            with pytest.raises(ValueError, match="num_blocks"):
                config.validate()

    def test_validate_positive_max_batch_size(self) -> None:
        """Test that max_batch_size must be positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EngineConfig(
                model=make_model_config(),
                model_path=tmpdir,
                max_batch_size=0,
            )
            with pytest.raises(ValueError, match="max_batch_size"):
                config.validate()

    def test_validate_model_path_exists(self) -> None:
        """Test that model_path must exist."""
        config = EngineConfig(
            model=make_model_config(),
            model_path="/nonexistent/path",
        )
        with pytest.raises(ValueError, match="does not exist"):
            config.validate()

    def test_validate_success(self) -> None:
        """Test that valid config passes validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EngineConfig(
                model=make_model_config(),
                model_path=tmpdir,
            )
            # Should not raise
            config.validate()


class TestEngineConfigDefaults:
    """Test EngineConfig default values."""

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EngineConfig(
                model=make_model_config(),
                model_path=tmpdir,
            )

            assert config.max_batch_size == 32
            assert config.num_blocks == 1024
            assert config.block_size == 16
            assert config.device == "cuda"
            assert config.dtype == torch.float16
            assert config.attention_backend == "auto"
            assert config.linear_backend == "pytorch"
            assert config.torch_compile is False
            assert config.warmup is False
