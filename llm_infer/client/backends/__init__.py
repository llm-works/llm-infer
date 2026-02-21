"""Backend implementations.

This module provides the backend base class and implementations.
For client creation, use Factory from llm_infer.client.
"""

from .base import Backend
from .openai import OpenAICompatibleBackend

__all__ = [
    "Backend",
    "OpenAICompatibleBackend",
]
