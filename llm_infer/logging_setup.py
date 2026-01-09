"""Third-party library logging configuration.

This module configures log levels for noisy third-party libraries.
Must be called BEFORE importing torch/transformers to be effective.
"""

import logging
import os
import warnings

# Map string level names to logging constants
_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_third_party_logging(
    torch_level: str = "warning",
    transformers_level: str = "error",
) -> None:
    """Configure third-party library log levels.

    Args:
        torch_level: Log level for PyTorch (torch._inductor, torch._dynamo, etc.)
        transformers_level: Log level for HuggingFace transformers

    Note:
        - Must be called BEFORE importing torch/transformers
        - HuggingFace tokenizers (Rust) prints to stdout and cannot be controlled here
    """
    # PyTorch C++ logging uses environment variables, not Python logging
    # TORCH_LOGS controls the internal C++ logging system
    if torch_level.lower() in ("warning", "error"):
        # Suppress inductor/dynamo verbose output
        os.environ.setdefault("TORCH_LOGS", "-all")
        os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")

    # Also set Python-side torch loggers for any Python-level logging
    torch_lvl = _LEVEL_MAP.get(torch_level.lower(), logging.WARNING)
    for name in ("torch._inductor", "torch._dynamo", "torch.distributed"):
        logging.getLogger(name).setLevel(torch_lvl)

    # HuggingFace transformers - use environment variable (must be set before import)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", transformers_level.lower())

    # Also suppress via warnings module (transformers sometimes uses warnings.warn)
    if transformers_level.lower() == "error":
        warnings.filterwarnings("ignore", module="transformers")

    # Suppress tokenizers parallelism warning
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Disable vLLM's dictConfig call which closes existing file handler streams
    # vLLM calls logging.config.dictConfig() which sets handler.stream = None
    # on existing FileHandlers, breaking file logging
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
