"""Model configuration and resolution.

This module provides:
- Configuration classes for model settings (ModelsConfig, ModelConfig, etc.)
- Model resolution logic (ModelResolver) for finding models in locations
- Metadata extraction from HuggingFace config.json (ModelMetadata)
"""

from .config import (
    ModelConfig,
    ModelsConfig,
    SelectionConfig,
    ThinkConfig,
    load_models_config,
)
from .metadata import ModelMetadata, get_model_metadata
from .resolver import ModelResolver, create_resolver

__all__ = [
    # Config classes
    "ModelsConfig",
    "ModelConfig",
    "SelectionConfig",
    "ThinkConfig",
    "load_models_config",
    # Metadata
    "ModelMetadata",
    "get_model_metadata",
    # Resolver
    "ModelResolver",
    "create_resolver",
]
