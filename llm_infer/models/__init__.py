"""Model configuration and resolution.

This module provides:
- Configuration classes for model settings (ModelsConfig, ModelConfig, etc.)
- Model resolution logic (ModelResolver) for finding models in locations
"""

from .config import (
    ModelConfig,
    ModelsConfig,
    SelectionConfig,
    ThinkConfig,
    load_models_config,
)
from .resolver import ModelResolver, create_resolver

__all__ = [
    # Config classes
    "ModelsConfig",
    "ModelConfig",
    "SelectionConfig",
    "ThinkConfig",
    "load_models_config",
    # Resolver
    "ModelResolver",
    "create_resolver",
]
