"""Backend implementations.

This module provides the backend base class and implementations.
For client creation, use Factory from llm_infer.client.
"""

from llm_infer.client.backends.base import Backend
from llm_infer.client.backends.openai import OpenAICompatibleBackend

__all__ = [
    "Backend",
    "OpenAICompatibleBackend",
]
