"""Model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TransformerConfig:
    """Configuration for a transformer model architecture.

    Parsed from HuggingFace config.json. Contains architectural details
    like layer count, dimensions, and quantization settings.
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    max_seq_len: int = 4096
    attention_bias: bool = False
    tie_word_embeddings: bool = False  # Share weights between embed_tokens and lm_head
    model_type: str | None = (
        None  # From HF config.json (e.g., "llama", "mistral", "qwen2")
    )

    # Granite-specific scaling multipliers (also used by other architectures if set)
    attention_multiplier: float | None = None  # None = use 1/sqrt(head_dim)
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0

    # Qwen3-specific: QK LayerNorm (RMSNorm on Q and K after projection)
    qk_norm: bool = False

    # Quantization config (None = not quantized, "awq" = AWQ 4-bit)
    quant_method: str | None = None
    quant_bits: int = 16  # 4 for AWQ
    quant_group_size: int = 128  # Typically 128 for AWQ

    @classmethod
    def from_name(cls, name: str) -> TransformerConfig:
        """Load config for a known model architecture."""
        # fmt: off
        configs = {
            "llama-7b": cls(num_layers=32, num_heads=32, num_kv_heads=32, head_dim=128, hidden_size=4096, intermediate_size=11008, vocab_size=32000, rope_theta=10000.0),
            "llama-3-8b": cls(num_layers=32, num_heads=32, num_kv_heads=8, head_dim=128, hidden_size=4096, intermediate_size=14336, vocab_size=128256, rope_theta=500000.0),
            "mistral-7b": cls(num_layers=32, num_heads=32, num_kv_heads=8, head_dim=128, hidden_size=4096, intermediate_size=14336, vocab_size=32000, rope_theta=10000.0),
        }
        # fmt: on
        if name not in configs:
            raise ValueError(
                f"Unknown model: {name}. Available: {list(configs.keys())}"
            )
        return configs[name]

    @classmethod
    def _load_hf_config(cls, model_path: str | Path) -> dict[str, Any]:
        """Load and parse HuggingFace config.json."""
        config_file = Path(model_path) / "config.json"
        if not config_file.exists():
            raise ValueError(f"No config.json found in {model_path}")
        with open(config_file) as f:
            result: dict[str, Any] = json.load(f)
            return result

    @classmethod
    def from_hf_config(cls, model_path: str | Path) -> TransformerConfig:
        """Load config from HuggingFace model directory."""
        from .architecture import get_config_defaults

        hf = cls._load_hf_config(model_path)
        num_heads, hidden_size = hf["num_attention_heads"], hf["hidden_size"]
        model_type = hf.get("model_type")
        dfl = get_config_defaults(model_type)
        return cls(
            num_layers=hf["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=hf.get("num_key_value_heads", num_heads),
            head_dim=hf.get("head_dim", hidden_size // num_heads),
            hidden_size=hidden_size,
            intermediate_size=hf["intermediate_size"],
            vocab_size=hf["vocab_size"],
            rope_theta=hf.get("rope_theta", 10000.0),
            rms_norm_eps=hf.get("rms_norm_eps", 1e-5),
            max_seq_len=hf.get("max_position_embeddings", 4096),
            attention_bias=hf.get("attention_bias", dfl.get("attention_bias", False)),
            tie_word_embeddings=hf.get("tie_word_embeddings", False),
            model_type=model_type,
            attention_multiplier=hf.get("attention_multiplier"),
            embedding_multiplier=hf.get("embedding_multiplier", 1.0),
            residual_multiplier=hf.get("residual_multiplier", 1.0),
            logits_scaling=hf.get("logits_scaling", 1.0),
            qk_norm=hf.get("qk_norm", dfl.get("qk_norm", False)),
            **cls._parse_quant_config(hf),
        )

    @classmethod
    def _parse_quant_config(cls, hf: dict) -> dict:
        """Parse quantization config from HuggingFace config.json."""
        if "quantization_config" not in hf:
            return {}

        qconfig = hf["quantization_config"]
        quant_method = qconfig.get("quant_method")

        if quant_method == "awq":
            return {
                "quant_method": "awq",
                "quant_bits": qconfig.get("bits", 4),
                "quant_group_size": qconfig.get("group_size", 128),
            }

        if quant_method == "fp8":
            return {
                "quant_method": "fp8",
                "quant_bits": 8,
                "quant_group_size": qconfig.get("weight_block_size", [128, 128])[0],
            }

        # Unknown quantization method - return empty (will use defaults)
        return {}
