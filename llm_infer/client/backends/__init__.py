"""Backend implementations.

This module provides the backend base class and implementations.
For client creation, use Factory from llm_infer.client.
"""

from . import embedding
from .base import Backend
from .context import BackendContext, RetryConfig
from .factory import BackendFactory
from .mixins import AsyncRequestTrackingMixin
from .provider import Provider, ProviderDetector
from .providers import AnthropicBackend, GeminiBackend, OpenAICompatibleBackend
from .vertex_native import NativeVertexBackend, NativeVertexFactory

__all__ = [
    "AnthropicBackend",
    "AsyncRequestTrackingMixin",
    "Backend",
    "BackendContext",
    "BackendFactory",
    "GeminiBackend",
    "NativeVertexBackend",
    "NativeVertexFactory",
    "OpenAICompatibleBackend",
    "Provider",
    "ProviderDetector",
    "RetryConfig",
    "embedding",
]
