"""Model architecture abstraction.

This module provides a unified abstraction for model architectures, consolidating
forward-pass behavior and tokenizer configuration into a single class hierarchy.

The design separates interface (ModelArchitecture ABC) from implementation
(LlamaArchitecture as default), allowing clean extension for new architectures.

Architecture Hierarchy:
    ModelArchitecture (ABC interface)
        └── LlamaArchitecture (default implementation)
                ├── MistralArchitecture (tokenizer fix)
                ├── GraniteArchitecture (scaling multipliers)
                └── Qwen3Architecture (QK LayerNorm)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from torch import Tensor

from ..tokenizer import TokenizerConfig

if TYPE_CHECKING:
    from torch import nn

    from .config import TransformerConfig

# Corrected Mistral pre-tokenizer regex
# The original regex in Mistral tokenizer files has a bug that causes incorrect tokenization.
# This pattern is the fixed version from HuggingFace's tokenization_utils_base.py
MISTRAL_PRETOKENIZER_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class ModelArchitecture(ABC):
    """Abstract interface for model architecture specifications.

    Defines the contract for architecture-specific behavior in the forward pass
    and tokenizer configuration. All architectures must implement these methods.
    """

    @abstractmethod
    def scale_embeddings(self, hidden: Tensor) -> Tensor:
        """Apply embedding scaling after token embedding lookup.

        Args:
            hidden: Embedded token representations [batch, seq_len, hidden_size]

        Returns:
            Scaled embeddings (may be unchanged for most architectures).
        """
        ...

    @abstractmethod
    def get_attention_scale(self) -> float:
        """Get the softmax scaling factor for attention.

        Standard transformers use 1/sqrt(head_dim). Some architectures
        (e.g., Granite) use custom scaling factors.

        Returns:
            Attention scaling factor.
        """
        ...

    @abstractmethod
    def apply_residual(self, residual: Tensor, output: Tensor) -> Tensor:
        """Apply residual connection.

        Standard: residual + output
        Some architectures (e.g., Granite) scale the output before adding.

        Args:
            residual: Input to the sublayer (attention or MLP)
            output: Output from the sublayer

        Returns:
            Combined residual connection result.
        """
        ...

    @abstractmethod
    def scale_logits(self, logits: Tensor) -> Tensor:
        """Apply output logit scaling.

        Some architectures (e.g., Granite) divide logits by a scaling factor.

        Args:
            logits: Raw output logits [batch, vocab_size]

        Returns:
            Scaled logits.
        """
        ...

    @abstractmethod
    def apply_qk_norm(
        self, q: Tensor, k: Tensor, layer: nn.ModuleDict
    ) -> tuple[Tensor, Tensor]:
        """Apply Q/K normalization if required by this architecture.

        Qwen3 applies RMSNorm to Q and K after projection but before RoPE.
        Most architectures (LLaMA, Mistral, etc.) skip this step.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            layer: ModuleDict containing layer weights (including q_norm, k_norm if present)

        Returns:
            Tuple of (normalized_q, normalized_k), or unchanged (q, k) for most architectures.
        """
        ...

    @abstractmethod
    def tokenizer_config(self) -> TokenizerConfig:
        """Get tokenizer configuration for this architecture.

        Returns:
            TokenizerConfig with architecture-specific settings.
        """
        ...


class LlamaArchitecture(ModelArchitecture):
    """Default implementation for LLaMA-compatible architectures.

    Provides standard transformer behavior:
    - No embedding scaling
    - Standard 1/sqrt(head_dim) attention scaling
    - Simple residual connections (residual + output)
    - No logit scaling

    Works for: LLaMA 1/2/3, Qwen, Qwen2, and other LLaMA-style models.
    """

    def __init__(self, config: TransformerConfig):
        """Initialize with model configuration.

        Args:
            config: Model configuration containing dimensions and parameters.
        """
        self.config = config

    def scale_embeddings(self, hidden: Tensor) -> Tensor:
        """No embedding scaling for LLaMA."""
        return hidden

    def get_attention_scale(self) -> float:
        """Standard 1/sqrt(head_dim) scaling."""
        return 1.0 / math.sqrt(self.config.head_dim)

    def apply_residual(self, residual: Tensor, output: Tensor) -> Tensor:
        """Standard residual connection."""
        return residual + output

    def scale_logits(self, logits: Tensor) -> Tensor:
        """No logit scaling for LLaMA."""
        return logits

    def apply_qk_norm(
        self, q: Tensor, k: Tensor, layer: nn.ModuleDict
    ) -> tuple[Tensor, Tensor]:
        """No QK normalization for LLaMA."""
        return q, k

    def tokenizer_config(self) -> TokenizerConfig:
        """Default tokenizer configuration."""
        return TokenizerConfig()


class MistralArchitecture(LlamaArchitecture):
    """Architecture for Mistral family models.

    Inherits all forward-pass behavior from LlamaArchitecture.
    Only overrides tokenizer config to fix pre-tokenizer regex bug.
    """

    def tokenizer_config(self) -> TokenizerConfig:
        """Tokenizer config with corrected pre-tokenizer pattern."""
        return TokenizerConfig(pre_tokenizer_pattern=MISTRAL_PRETOKENIZER_PATTERN)


class Qwen3Architecture(LlamaArchitecture):
    """Architecture for Qwen3 models.

    Qwen3 uses QK LayerNorm: RMSNorm applied to Q and K after projection
    but before RoPE. All other forward-pass behavior is standard LLaMA.
    """

    def apply_qk_norm(
        self, q: Tensor, k: Tensor, layer: nn.ModuleDict
    ) -> tuple[Tensor, Tensor]:
        """Apply RMSNorm to Q and K tensors."""
        return layer["q_norm"](q), layer["k_norm"](k)


class GraniteArchitecture(LlamaArchitecture):
    """Architecture for IBM Granite 3.x models.

    Granite uses custom scaling multipliers throughout the forward pass:
    - embedding_multiplier: Scales embeddings after token lookup
    - attention_multiplier: Replaces standard 1/sqrt(head_dim) scaling
    - residual_multiplier: Scales sublayer output before residual add
    - logits_scaling: Divides final logits

    These values are read from TransformerConfig (parsed from HF config.json).
    """

    def scale_embeddings(self, hidden: Tensor) -> Tensor:
        """Apply Granite embedding multiplier."""
        if self.config.embedding_multiplier != 1.0:
            return hidden * self.config.embedding_multiplier
        return hidden

    def get_attention_scale(self) -> float:
        """Use Granite attention_multiplier if set, else default."""
        if self.config.attention_multiplier is not None:
            return self.config.attention_multiplier
        return super().get_attention_scale()

    def apply_residual(self, residual: Tensor, output: Tensor) -> Tensor:
        """Apply Granite residual multiplier."""
        if self.config.residual_multiplier != 1.0:
            return residual + output * self.config.residual_multiplier
        return super().apply_residual(residual, output)

    def scale_logits(self, logits: Tensor) -> Tensor:
        """Apply Granite logits scaling (division)."""
        if self.config.logits_scaling != 1.0:
            return logits / self.config.logits_scaling
        return logits


# Registry mapping model_type to architecture class
# model_type comes from config.json's "model_type" field
ARCHITECTURES: dict[str, type[LlamaArchitecture]] = {
    "mistral": MistralArchitecture,
    "qwen": LlamaArchitecture,
    "qwen2": LlamaArchitecture,
    "qwen3": Qwen3Architecture,
    "granite": GraniteArchitecture,
}

# Model-specific config defaults
# Maps model_type to config field overrides when HF config doesn't specify them
CONFIG_DEFAULTS: dict[str, dict] = {
    "qwen": {"attention_bias": True},
    "qwen2": {"attention_bias": True},
    "qwen3": {"attention_bias": True, "qk_norm": True},
}


def get_config_defaults(model_type: str | None) -> dict:
    """Get model-specific config defaults.

    Args:
        model_type: Model type from HF config.json (e.g., "llama", "qwen3").

    Returns:
        Dict of config field defaults for this model type.
        Empty dict for unknown types (use standard defaults).
    """
    return CONFIG_DEFAULTS.get(model_type or "", {})


def get_architecture(lg: Any, config: TransformerConfig) -> LlamaArchitecture:
    """Get architecture instance for a model configuration.

    Args:
        lg: Logger for warnings.
        config: Model configuration with model_type and parameters.

    Returns:
        LlamaArchitecture (or subclass) instance appropriate for the model type.
        Falls back to LlamaArchitecture for unknown types.
    """
    model_type = config.model_type

    if model_type is None:
        lg.warning(
            "model_type not found in config.json, defaulting to LlamaArchitecture"
        )
        return LlamaArchitecture(config)

    if model_type in ARCHITECTURES:
        return ARCHITECTURES[model_type](config)

    # Unknown model_type - may be LLaMA-compatible or may need custom architecture
    known_types = ", ".join(sorted(ARCHITECTURES.keys()))
    lg.warning(
        f"unknown model_type '{model_type}', defaulting to LlamaArchitecture",
        extra={"known_types": known_types},
    )
    return LlamaArchitecture(config)
