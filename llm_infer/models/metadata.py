"""Model metadata extraction from HuggingFace config.json.

This module provides utilities to extract quantization and precision metadata
from HuggingFace model configurations, enabling:
- llm-learn training: auto-set fp16/bf16 based on quantization
- Inference: select appropriate dtype
- Memory estimation: factor in quantization bits
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .resolver import ModelResolver


@dataclass
class ModelMetadata:
    """Model metadata extracted from HuggingFace config.json.

    Attributes:
        quantization: Quantization method ("bitsandbytes", "gptq", "awq", "fp8", None).
        quantization_bits: Bit width for quantization (4 or 8).
        torch_dtype: Precision from config ("bfloat16", "float16", "float32").
        num_params_b: Approximate parameter count in billions, or None if unavailable.
    """

    quantization: str | None = None
    quantization_bits: int | None = None
    torch_dtype: str | None = None
    num_params_b: float | None = None

    @property
    def is_quantized(self) -> bool:
        """Whether the model is pre-quantized."""
        return self.quantization is not None

    @property
    def recommended_fp16(self) -> bool:
        """Whether FP16 AMP training is recommended.

        AMP is incompatible with pre-quantized models (e.g., BNB 4-bit),
        so this returns False for quantized models.
        """
        return not self.is_quantized

    @property
    def recommended_bf16(self) -> bool:
        """Whether BF16 training is recommended.

        BF16 is also incompatible with pre-quantized models,
        so this returns False for quantized models.
        """
        return not self.is_quantized


def get_model_metadata(
    *,
    path: Path | str | None = None,
    name: str | None = None,
    resolver: ModelResolver | None = None,
) -> ModelMetadata:
    """Extract metadata from a HuggingFace model directory.

    Reads config.json and extracts quantization and precision information.
    Supports multiple quantization methods: bitsandbytes, gptq, awq, fp8.

    Args:
        path: Direct path to model directory containing config.json.
        name: Model name to resolve via resolver.
        resolver: ModelResolver instance for name resolution.

    Returns:
        ModelMetadata with quantization and precision info.

    Raises:
        ValueError: If neither path nor name provided, or name without resolver.
        FileNotFoundError: If config.json doesn't exist or model not found.
    """
    if path is None and name is None:
        raise ValueError("Either path or name must be provided")
    if name is not None and resolver is None:
        raise ValueError("resolver required when using name")

    if path is None:
        # At this point, name must be non-None (validated above) and resolver is non-None
        assert name is not None and resolver is not None
        resolved = resolver.find_by_name(name)
        if resolved is None:
            raise FileNotFoundError(f"Model not found: {name}")
        path = resolved

    config_path = Path(path) / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    return _parse_config(config)


def _parse_config(config: dict) -> ModelMetadata:
    """Parse HuggingFace config.json into ModelMetadata."""
    quant_config = config.get("quantization_config") or {}
    if not isinstance(quant_config, dict):
        quant_config = {}
    quant_method = quant_config.get("quant_method")

    bits = _extract_bits(quant_method, quant_config)

    return ModelMetadata(
        quantization=quant_method,
        quantization_bits=bits,
        torch_dtype=config.get("torch_dtype"),
        num_params_b=_compute_num_params_b(config),
    )


def _compute_num_params_b(config: dict, *, detailed: bool = True) -> float | None:
    """Compute approximate parameter count in billions from config.json.

    Args:
        config: HuggingFace config.json as dict.
        detailed: If True (default), account for GQA, SwiGLU, layer norms,
            and lm_head. If False, use simplified formula.

    Returns None if required fields are missing.
    """
    hidden_size: int | None = config.get("hidden_size")
    intermediate_size: int | None = config.get("intermediate_size")
    num_layers: int | None = config.get("num_hidden_layers")
    vocab_size: int | None = config.get("vocab_size")

    if hidden_size is None or intermediate_size is None:
        return None
    if num_layers is None or vocab_size is None:
        return None

    total_params = _compute_core_params(
        config, num_layers, hidden_size, intermediate_size, vocab_size, detailed
    )
    if detailed:
        total_params += _compute_auxiliary_params(
            config, num_layers, hidden_size, vocab_size
        )

    return round(total_params / 1e9, 2)


def _compute_core_params(
    config: dict,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    detailed: bool,
) -> int:
    """Compute embedding, attention, and FFN parameters."""
    embedding = vocab_size * hidden_size

    if detailed:
        attention = _compute_attention_params(config, num_layers, hidden_size)
    else:
        attention = num_layers * (4 * hidden_size * hidden_size)

    # SwiGLU uses 3 projections, standard uses 2
    ffn_multiplier = 3 if (detailed and _is_swiglu(config)) else 2
    ffn = num_layers * (ffn_multiplier * hidden_size * intermediate_size)

    return embedding + attention + ffn


def _compute_auxiliary_params(
    config: dict, num_layers: int, hidden_size: int, vocab_size: int
) -> int:
    """Compute layer norm and lm_head parameters."""
    # Layer norms: 2 per layer (attn + ffn) + 1 final
    layer_norms = (2 * num_layers + 1) * hidden_size

    # Output projection (lm_head) if not tied with embeddings
    lm_head = (
        0 if config.get("tie_word_embeddings", False) else vocab_size * hidden_size
    )

    return layer_norms + lm_head


def _compute_attention_params(config: dict, num_layers: int, hidden_size: int) -> int:
    """Compute attention layer parameters, accounting for GQA if present."""
    num_heads: int | None = config.get("num_attention_heads")
    num_kv_heads: int | None = config.get("num_key_value_heads")

    if num_heads is None or num_kv_heads is None or num_heads == num_kv_heads:
        # Standard MHA: Q, K, V, O all same size
        return num_layers * (4 * hidden_size * hidden_size)

    # GQA: K/V projections are smaller
    head_dim = hidden_size // num_heads
    kv_dim = head_dim * num_kv_heads

    q_proj = hidden_size * hidden_size
    k_proj = hidden_size * kv_dim
    v_proj = hidden_size * kv_dim
    o_proj = hidden_size * hidden_size

    return num_layers * (q_proj + k_proj + v_proj + o_proj)


def _is_swiglu(config: dict) -> bool:
    """Check if model uses SwiGLU activation (3 FFN projections instead of 2)."""
    hidden_act = config.get("hidden_act", "")
    return "silu" in hidden_act.lower() or "swiglu" in hidden_act.lower()


def _extract_bits(quant_method: str | None, quant_config: dict) -> int | None:
    """Extract quantization bits based on method.

    Different quantization methods store bit information differently:
    - bitsandbytes: load_in_4bit/load_in_8bit or _load_in_4bit/_load_in_8bit
    - gptq/awq/fp8: bits field directly
    """
    if quant_method == "bitsandbytes":
        if quant_config.get("load_in_4bit") or quant_config.get("_load_in_4bit"):
            return 4
        if quant_config.get("load_in_8bit") or quant_config.get("_load_in_8bit"):
            return 8
        return None
    if quant_method in ("gptq", "awq", "fp8"):
        return quant_config.get("bits")
    return None
