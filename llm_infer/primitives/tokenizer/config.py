"""Tokenizer configuration."""

from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Abstract tokenizer configuration - no backend-specific details.

    This dataclass defines granular configuration points for tokenizer behavior.
    Each field represents a specific, well-defined aspect of tokenization that
    can be influenced by model policies.

    Backend implementations (e.g., HuggingFaceTokenizer) map these abstract
    config options to their specific internals.
    """

    pre_tokenizer_pattern: str | None = None
    """Override pre-tokenizer regex pattern.

    Some models (e.g., Mistral) ship with buggy pre-tokenizer patterns.
    This allows injecting a corrected pattern without modifying model files.
    """

    # Future config points:
    # add_bos_token: bool = True
    # add_eos_token: bool = False
    # clean_up_tokenization_spaces: bool = True
