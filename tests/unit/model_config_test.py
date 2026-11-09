# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Unit tests for model configuration."""

import pytest

from llm_infer.engines.native.model.config import TransformerConfig


class TestTransformerConfigFromName:
    """Test TransformerConfig.from_name factory."""

    def test_llama_7b(self) -> None:
        """Test loading llama-7b config."""
        config = TransformerConfig.from_name("llama-7b")
        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.num_kv_heads == 32
        assert config.hidden_size == 4096
        assert config.vocab_size == 32000

    def test_llama_3_8b(self) -> None:
        """Test loading llama-3-8b config."""
        config = TransformerConfig.from_name("llama-3-8b")
        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.num_kv_heads == 8  # GQA
        assert config.hidden_size == 4096
        assert config.vocab_size == 128256
        assert config.rope_theta == 500000.0

    def test_mistral_7b(self) -> None:
        """Test loading mistral-7b config."""
        config = TransformerConfig.from_name("mistral-7b")
        assert config.num_layers == 32
        assert config.num_kv_heads == 8  # GQA

    def test_unknown_model_raises(self) -> None:
        """Test that unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            TransformerConfig.from_name("unknown-model")


class TestTransformerConfigParseQuantConfig:
    """Test quantization config parsing."""

    def test_no_quant_config(self) -> None:
        """Test parsing HF config without quantization."""
        result = TransformerConfig._parse_quant_config({})
        assert result == {}

    def test_awq_quant_config(self) -> None:
        """Test parsing AWQ quantization config."""
        hf_config = {
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
            }
        }
        result = TransformerConfig._parse_quant_config(hf_config)
        assert result["quant_method"] == "awq"
        assert result["quant_bits"] == 4
        assert result["quant_group_size"] == 128

    def test_awq_defaults(self) -> None:
        """Test AWQ config uses defaults when fields missing."""
        hf_config = {
            "quantization_config": {
                "quant_method": "awq",
            }
        }
        result = TransformerConfig._parse_quant_config(hf_config)
        assert result["quant_bits"] == 4  # default
        assert result["quant_group_size"] == 128  # default

    def test_fp8_quant_config(self) -> None:
        """Test parsing FP8 quantization config."""
        hf_config = {
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
        }
        result = TransformerConfig._parse_quant_config(hf_config)
        assert result["quant_method"] == "fp8"
        assert result["quant_bits"] == 8
        assert result["quant_group_size"] == 128

    def test_unknown_quant_method(self) -> None:
        """Test that unknown quant method returns empty dict."""
        hf_config = {
            "quantization_config": {
                "quant_method": "unknown",
            }
        }
        result = TransformerConfig._parse_quant_config(hf_config)
        assert result == {}


class TestTransformerConfigDefaults:
    """Test TransformerConfig default values."""

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        config = TransformerConfig(
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
        )
        assert config.rope_theta == 10000.0
        assert config.rms_norm_eps == 1e-5
        assert config.max_seq_len == 4096
        assert config.attention_bias is False
        assert config.tie_word_embeddings is False
        assert config.quant_method is None
        assert config.embedding_multiplier == 1.0
        assert config.residual_multiplier == 1.0
        assert config.logits_scaling == 1.0
        assert config.qk_norm is False
