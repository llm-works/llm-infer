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
    """

    quantization: str | None = None
    quantization_bits: int | None = None
    torch_dtype: str | None = None

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
    quant_config = config.get("quantization_config", {})
    quant_method = quant_config.get("quant_method")

    bits = _extract_bits(quant_method, quant_config)

    return ModelMetadata(
        quantization=quant_method,
        quantization_bits=bits,
        torch_dtype=config.get("torch_dtype"),
    )


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
