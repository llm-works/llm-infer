"""Unit tests for model architecture."""

import math
from unittest.mock import MagicMock

import pytest
import torch

from llm_infer.pipelines.model.architecture import (
    ARCHITECTURES,
    GraniteArchitecture,
    LlamaArchitecture,
    MistralArchitecture,
    Qwen3Architecture,
    get_architecture,
    get_config_defaults,
)
from llm_infer.pipelines.model.config import ModelConfig

pytestmark = pytest.mark.unit


def make_config(**overrides) -> ModelConfig:
    """Create a test ModelConfig with defaults."""
    defaults = {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 32000,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestLlamaArchitecture:
    """Test LlamaArchitecture (default implementation)."""

    def test_scale_embeddings_unchanged(self) -> None:
        """Test that embeddings are not scaled."""
        config = make_config()
        arch = LlamaArchitecture(config)
        hidden = torch.randn(2, 10, 4096)
        result = arch.scale_embeddings(hidden)
        assert torch.equal(result, hidden)

    def test_attention_scale(self) -> None:
        """Test standard 1/sqrt(head_dim) scaling."""
        config = make_config(head_dim=128)
        arch = LlamaArchitecture(config)
        expected = 1.0 / math.sqrt(128)
        assert arch.get_attention_scale() == expected

    def test_apply_residual(self) -> None:
        """Test standard residual connection."""
        config = make_config()
        arch = LlamaArchitecture(config)
        residual = torch.tensor([1.0, 2.0, 3.0])
        output = torch.tensor([0.5, 0.5, 0.5])
        result = arch.apply_residual(residual, output)
        expected = torch.tensor([1.5, 2.5, 3.5])
        assert torch.allclose(result, expected)

    def test_scale_logits_unchanged(self) -> None:
        """Test that logits are not scaled."""
        config = make_config()
        arch = LlamaArchitecture(config)
        logits = torch.randn(2, 32000)
        result = arch.scale_logits(logits)
        assert torch.equal(result, logits)

    def test_apply_qk_norm_passthrough(self) -> None:
        """Test that Q/K are passed through unchanged."""
        config = make_config()
        arch = LlamaArchitecture(config)
        q = torch.randn(2, 10, 32, 128)
        k = torch.randn(2, 10, 8, 128)
        layer = MagicMock()
        q_out, k_out = arch.apply_qk_norm(q, k, layer)
        assert torch.equal(q_out, q)
        assert torch.equal(k_out, k)

    def test_tokenizer_config(self) -> None:
        """Test default tokenizer config."""
        config = make_config()
        arch = LlamaArchitecture(config)
        tok_config = arch.tokenizer_config()
        assert tok_config.pre_tokenizer_pattern is None


class TestMistralArchitecture:
    """Test MistralArchitecture."""

    def test_tokenizer_has_pretokenizer_pattern(self) -> None:
        """Test that Mistral has corrected pre-tokenizer pattern."""
        config = make_config()
        arch = MistralArchitecture(config)
        tok_config = arch.tokenizer_config()
        assert tok_config.pre_tokenizer_pattern is not None
        assert "\\p{L}" in tok_config.pre_tokenizer_pattern


class TestGraniteArchitecture:
    """Test GraniteArchitecture with custom scaling."""

    def test_scale_embeddings_with_multiplier(self) -> None:
        """Test embedding scaling with multiplier."""
        config = make_config(embedding_multiplier=2.0)
        arch = GraniteArchitecture(config)
        hidden = torch.ones(2, 10, 4096)
        result = arch.scale_embeddings(hidden)
        assert torch.allclose(result, torch.full_like(hidden, 2.0))

    def test_scale_embeddings_unity_multiplier(self) -> None:
        """Test embedding scaling with unity multiplier (passthrough)."""
        config = make_config(embedding_multiplier=1.0)
        arch = GraniteArchitecture(config)
        hidden = torch.randn(2, 10, 4096)
        result = arch.scale_embeddings(hidden)
        assert torch.equal(result, hidden)

    def test_attention_scale_custom(self) -> None:
        """Test custom attention multiplier."""
        config = make_config(attention_multiplier=0.125)
        arch = GraniteArchitecture(config)
        assert arch.get_attention_scale() == 0.125

    def test_attention_scale_fallback(self) -> None:
        """Test attention scale falls back to default when not set."""
        config = make_config(attention_multiplier=None, head_dim=128)
        arch = GraniteArchitecture(config)
        expected = 1.0 / math.sqrt(128)
        assert arch.get_attention_scale() == expected

    def test_apply_residual_with_multiplier(self) -> None:
        """Test residual connection with multiplier."""
        config = make_config(residual_multiplier=0.5)
        arch = GraniteArchitecture(config)
        residual = torch.tensor([1.0, 2.0, 3.0])
        output = torch.tensor([2.0, 2.0, 2.0])
        result = arch.apply_residual(residual, output)
        # residual + output * 0.5 = [1, 2, 3] + [1, 1, 1] = [2, 3, 4]
        expected = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(result, expected)

    def test_scale_logits_with_scaling(self) -> None:
        """Test logit scaling (division)."""
        config = make_config(logits_scaling=2.0)
        arch = GraniteArchitecture(config)
        logits = torch.tensor([4.0, 8.0, 16.0])
        result = arch.scale_logits(logits)
        expected = torch.tensor([2.0, 4.0, 8.0])
        assert torch.allclose(result, expected)


class TestGetConfigDefaults:
    """Test get_config_defaults function."""

    def test_qwen_defaults(self) -> None:
        """Test Qwen has attention_bias default."""
        defaults = get_config_defaults("qwen")
        assert defaults["attention_bias"] is True

    def test_qwen3_defaults(self) -> None:
        """Test Qwen3 has qk_norm and attention_bias defaults."""
        defaults = get_config_defaults("qwen3")
        assert defaults["attention_bias"] is True
        assert defaults["qk_norm"] is True

    def test_unknown_model_empty_defaults(self) -> None:
        """Test unknown model type returns empty dict."""
        defaults = get_config_defaults("unknown")
        assert defaults == {}

    def test_none_model_empty_defaults(self) -> None:
        """Test None model type returns empty dict."""
        defaults = get_config_defaults(None)
        assert defaults == {}


class TestGetArchitecture:
    """Test get_architecture function."""

    def test_returns_llama_for_none_type(self) -> None:
        """Test that None model_type returns LlamaArchitecture."""
        config = make_config(model_type=None)
        lg = MagicMock()
        arch = get_architecture(lg, config)
        assert isinstance(arch, LlamaArchitecture)
        lg.warning.assert_called()

    def test_returns_mistral_architecture(self) -> None:
        """Test that mistral type returns MistralArchitecture."""
        config = make_config(model_type="mistral")
        lg = MagicMock()
        arch = get_architecture(lg, config)
        assert isinstance(arch, MistralArchitecture)

    def test_returns_granite_architecture(self) -> None:
        """Test that granite type returns GraniteArchitecture."""
        config = make_config(model_type="granite")
        lg = MagicMock()
        arch = get_architecture(lg, config)
        assert isinstance(arch, GraniteArchitecture)

    def test_returns_qwen3_architecture(self) -> None:
        """Test that qwen3 type returns Qwen3Architecture."""
        config = make_config(model_type="qwen3")
        lg = MagicMock()
        arch = get_architecture(lg, config)
        assert isinstance(arch, Qwen3Architecture)

    def test_unknown_type_falls_back_to_llama(self) -> None:
        """Test that unknown model_type falls back to LlamaArchitecture."""
        config = make_config(model_type="unknown_model")
        lg = MagicMock()
        arch = get_architecture(lg, config)
        assert isinstance(arch, LlamaArchitecture)
        lg.warning.assert_called()


class TestQwen3Architecture:
    """Test Qwen3Architecture with QK LayerNorm."""

    def test_apply_qk_norm_calls_norms(self) -> None:
        """Test that QK norm applies layer norms."""
        config = make_config(qk_norm=True)
        arch = Qwen3Architecture(config)

        # Create mock tensors and layer
        q = torch.randn(2, 10, 32, 128)
        k = torch.randn(2, 10, 8, 128)

        q_norm = MagicMock(return_value=q * 0.5)
        k_norm = MagicMock(return_value=k * 0.5)
        layer = {"q_norm": q_norm, "k_norm": k_norm}

        q_out, k_out = arch.apply_qk_norm(q, k, layer)

        q_norm.assert_called_once_with(q)
        k_norm.assert_called_once_with(k)
        assert torch.allclose(q_out, q * 0.5)
        assert torch.allclose(k_out, k * 0.5)


class TestArchitectureRegistry:
    """Test ARCHITECTURES registry."""

    def test_known_architectures(self) -> None:
        """Test that known architectures are in registry."""
        assert "mistral" in ARCHITECTURES
        assert "granite" in ARCHITECTURES
        assert "qwen3" in ARCHITECTURES
        assert "qwen" in ARCHITECTURES
        assert "qwen2" in ARCHITECTURES

    def test_registry_values_are_classes(self) -> None:
        """Test that registry values are architecture classes."""
        for name, arch_cls in ARCHITECTURES.items():
            assert issubclass(arch_cls, LlamaArchitecture)


class TestGraniteEdgeCases:
    """Test GraniteArchitecture edge cases."""

    def test_residual_unity_multiplier(self) -> None:
        """Test residual with unity multiplier uses super()."""
        config = make_config(residual_multiplier=1.0)
        arch = GraniteArchitecture(config)
        residual = torch.tensor([1.0, 2.0])
        output = torch.tensor([0.5, 0.5])
        result = arch.apply_residual(residual, output)
        expected = torch.tensor([1.5, 2.5])
        assert torch.allclose(result, expected)

    def test_logits_unity_scaling(self) -> None:
        """Test logits with unity scaling (passthrough)."""
        config = make_config(logits_scaling=1.0)
        arch = GraniteArchitecture(config)
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = arch.scale_logits(logits)
        assert torch.equal(result, logits)
