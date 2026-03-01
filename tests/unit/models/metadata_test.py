"""Unit tests for model metadata extraction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_infer.models import ModelMetadata, get_model_metadata

pytestmark = pytest.mark.unit


class TestModelMetadataProperties:
    """Test ModelMetadata dataclass properties."""

    def test_is_quantized_true_for_quantized(self) -> None:
        """Test is_quantized returns True when quantization is set."""
        meta = ModelMetadata(quantization="bitsandbytes", quantization_bits=4)
        assert meta.is_quantized is True

    def test_is_quantized_false_for_unquantized(self) -> None:
        """Test is_quantized returns False when quantization is None."""
        meta = ModelMetadata(quantization=None)
        assert meta.is_quantized is False

    def test_recommended_fp16_false_for_quantized(self) -> None:
        """Test recommended_fp16 returns False for quantized models."""
        meta = ModelMetadata(quantization="bitsandbytes", quantization_bits=4)
        assert meta.recommended_fp16 is False

    def test_recommended_fp16_true_for_unquantized(self) -> None:
        """Test recommended_fp16 returns True for unquantized models."""
        meta = ModelMetadata(quantization=None)
        assert meta.recommended_fp16 is True

    def test_recommended_bf16_false_for_quantized(self) -> None:
        """Test recommended_bf16 returns False for quantized models."""
        meta = ModelMetadata(quantization="gptq", quantization_bits=4)
        assert meta.recommended_bf16 is False

    def test_recommended_bf16_true_for_unquantized(self) -> None:
        """Test recommended_bf16 returns True for unquantized models."""
        meta = ModelMetadata(quantization=None, torch_dtype="bfloat16")
        assert meta.recommended_bf16 is True


class TestGetModelMetadataBitsAndBytes:
    """Test get_model_metadata with bitsandbytes quantization."""

    def test_bnb_4bit_load_in_4bit(self) -> None:
        """Test BNB 4-bit detection via load_in_4bit field."""
        config = {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization == "bitsandbytes"
            assert meta.quantization_bits == 4
            assert meta.torch_dtype == "bfloat16"
            assert meta.is_quantized is True
            assert meta.recommended_fp16 is False

    def test_bnb_4bit_underscore_prefix(self) -> None:
        """Test BNB 4-bit detection via _load_in_4bit field."""
        config = {
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "_load_in_4bit": True,
            },
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization_bits == 4

    def test_bnb_8bit(self) -> None:
        """Test BNB 8-bit detection."""
        config = {
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_8bit": True,
            },
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization == "bitsandbytes"
            assert meta.quantization_bits == 8


class TestGetModelMetadataGPTQ:
    """Test get_model_metadata with GPTQ quantization."""

    def test_gptq_4bit(self) -> None:
        """Test GPTQ 4-bit detection."""
        config = {
            "torch_dtype": "float16",
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
            },
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization == "gptq"
            assert meta.quantization_bits == 4
            assert meta.torch_dtype == "float16"

    def test_gptq_8bit(self) -> None:
        """Test GPTQ 8-bit detection."""
        config = {
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 8,
            },
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization == "gptq"
            assert meta.quantization_bits == 8


class TestGetModelMetadataAWQ:
    """Test get_model_metadata with AWQ quantization."""

    def test_awq_4bit(self) -> None:
        """Test AWQ 4-bit detection."""
        config = {
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
            },
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization == "awq"
            assert meta.quantization_bits == 4


class TestGetModelMetadataFP8:
    """Test get_model_metadata with FP8 quantization."""

    def test_fp8(self) -> None:
        """Test FP8 detection."""
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "bits": 8,
            },
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization == "fp8"
            assert meta.quantization_bits == 8


class TestGetModelMetadataUnquantized:
    """Test get_model_metadata with non-quantized models."""

    def test_bfloat16_model(self) -> None:
        """Test non-quantized bfloat16 model."""
        config = {
            "torch_dtype": "bfloat16",
            "model_type": "llama",
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization is None
            assert meta.quantization_bits is None
            assert meta.torch_dtype == "bfloat16"
            assert meta.is_quantized is False
            assert meta.recommended_fp16 is True
            assert meta.recommended_bf16 is True

    def test_float16_model(self) -> None:
        """Test non-quantized float16 model."""
        config = {
            "torch_dtype": "float16",
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.torch_dtype == "float16"
            assert meta.is_quantized is False

    def test_no_torch_dtype(self) -> None:
        """Test model without torch_dtype field."""
        config = {"model_type": "llama"}
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.torch_dtype is None

    def test_null_quantization_config(self) -> None:
        """Test model with null quantization_config (non-dict value)."""
        config = {
            "torch_dtype": "bfloat16",
            "quantization_config": None,
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.quantization is None
            assert meta.quantization_bits is None
            assert meta.is_quantized is False


class TestNumParamsB:
    """Test num_params_b parameter count estimation."""

    def test_standard_mha_model(self) -> None:
        """Test parameter count for standard MHA model."""
        # Simplified Llama-like config: 32 layers, 4096 hidden, 11008 intermediate
        config = {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,  # MHA (not GQA)
            "hidden_act": "silu",
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            # Should be around 7B params
            assert meta.num_params_b is not None
            assert 6.5 < meta.num_params_b < 7.5

    def test_gqa_model(self) -> None:
        """Test parameter count accounts for GQA (grouped query attention)."""
        # Llama 2 70B-like with GQA: 8 KV heads vs 64 Q heads
        config = {
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "vocab_size": 32000,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,  # GQA
            "hidden_act": "silu",
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            # GQA reduces K/V projection params
            assert meta.num_params_b is not None
            # ~65B with GQA savings
            assert 60 < meta.num_params_b < 75

    def test_swiglu_detection(self) -> None:
        """Test SwiGLU is detected and uses 3x FFN projections."""
        config = {
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 12,
            "vocab_size": 32000,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "hidden_act": "silu",  # SwiGLU indicator
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.num_params_b is not None
            # Just verify it computes without error
            assert meta.num_params_b > 0

    def test_missing_required_fields_returns_none(self) -> None:
        """Test that missing required fields returns None for num_params_b."""
        # Missing hidden_size
        config = {
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.num_params_b is None

    def test_null_hidden_act_handled(self) -> None:
        """Test that null hidden_act doesn't crash (fixed bug)."""
        config = {
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 12,
            "vocab_size": 32000,
            "hidden_act": None,  # Explicitly null
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            # Should not crash, and should compute params (non-SwiGLU path)
            assert meta.num_params_b is not None

    def test_missing_hidden_act_uses_default(self) -> None:
        """Test that missing hidden_act field doesn't crash."""
        config = {
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_hidden_layers": 12,
            "vocab_size": 32000,
            # hidden_act not present
        }
        with TempModelDir(config) as path:
            meta = get_model_metadata(path=path)
            assert meta.num_params_b is not None


class TestGetModelMetadataErrors:
    """Test get_model_metadata error cases."""

    def test_no_path_or_name_raises(self) -> None:
        """Test that neither path nor name raises ValueError."""
        with pytest.raises(ValueError, match="Either path or name must be provided"):
            get_model_metadata()

    def test_name_without_resolver_raises(self) -> None:
        """Test that name without resolver raises ValueError."""
        with pytest.raises(ValueError, match="resolver required when using name"):
            get_model_metadata(name="some-model")

    def test_missing_config_json_raises(self) -> None:
        """Test that missing config.json raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                get_model_metadata(path=tmpdir)

    def test_model_not_found_raises(self) -> None:
        """Test that model not found via resolver raises FileNotFoundError."""
        resolver = MagicMock()
        resolver.find_by_name.return_value = None
        with pytest.raises(FileNotFoundError, match="Model not found: missing-model"):
            get_model_metadata(name="missing-model", resolver=resolver)


class TestGetModelMetadataWithResolver:
    """Test get_model_metadata with ModelResolver."""

    def test_resolve_by_name(self) -> None:
        """Test resolving model by name via resolver."""
        config = {
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
            },
        }
        with TempModelDir(config) as path:
            resolver = MagicMock()
            resolver.find_by_name.return_value = path

            meta = get_model_metadata(name="test-model", resolver=resolver)

            resolver.find_by_name.assert_called_once_with("test-model")
            assert meta.quantization == "bitsandbytes"
            assert meta.quantization_bits == 4


# --- Test utilities ---


class TempModelDir:
    """Context manager for creating a temporary model directory with config.json."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._tmpdir = tempfile.TemporaryDirectory()
        path = Path(self._tmpdir.name)
        config_path = path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f)
        return path

    def __exit__(self, *args: object) -> None:
        if self._tmpdir:
            self._tmpdir.cleanup()
