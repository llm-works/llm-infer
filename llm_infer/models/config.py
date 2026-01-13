"""Model configuration classes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ThinkConfig:
    """Configuration for thinking mode behavior."""

    default: bool = False
    enable_suffix: str | None = None
    disable_suffix: str | None = None
    system_prompt: str | None = None
    tags_open: list[str] = field(default_factory=lambda: ["<think>", "<thinking>"])
    tags_close: list[str] = field(default_factory=lambda: ["</think>", "</thinking>"])

    @classmethod
    def from_dict(cls, data: dict) -> "ThinkConfig":
        """Create ThinkConfig from dict (YAML section)."""
        tags = data.get("tags", {})
        return cls(
            default=data.get("default", False),
            enable_suffix=data.get("enable_suffix"),
            disable_suffix=data.get("disable_suffix"),
            system_prompt=data.get("system_prompt"),
            tags_open=tags.get("open", ["<think>", "<thinking>"]),
            tags_close=tags.get("close", ["</think>", "</thinking>"]),
        )


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    task: str | None = None  # "generate" or "embed" - overrides engines.vllm.task
    max_model_len: int | None = None  # Override engines.vllm.max_model_len
    system_prompt: str | None = None
    think: ThinkConfig = field(default_factory=ThinkConfig)

    # Sentinel to distinguish "not set" from "explicitly set to None"
    _max_model_len_set: bool = field(default=False, repr=False)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "ModelConfig":
        """Create ModelConfig from dict (YAML section)."""
        think_data = data.get("think") or {}
        return cls(
            name=name,
            task=data.get("task"),
            max_model_len=data.get("max_model_len"),
            _max_model_len_set="max_model_len" in data,
            system_prompt=data.get("system_prompt"),
            think=ThinkConfig.from_dict(think_data),
        )


@dataclass
class SelectionConfig:
    """Selection config for a task type (generate or embed)."""

    path: str | None = None  # Path to selection file
    default: str | None = None  # Fallback model name

    @classmethod
    def from_dict(cls, data: dict) -> "SelectionConfig":
        """Create SelectionConfig from dict."""
        raw_path = data.get("path")
        return cls(
            path=str(raw_path) if raw_path else None,
            default=data.get("default"),
        )


@dataclass
class ModelsConfig:
    """Container for all model configurations.

    Unified config for model locations, selection, and per-model settings.
    Parsed from the models section of llm-infer.yaml (via !include models.yaml).
    """

    # Per-model configurations
    models: dict[str, ModelConfig] = field(default_factory=dict)
    defaults: ModelConfig = field(default_factory=lambda: ModelConfig(name="defaults"))

    # Model locations - directories to search for models
    locations: list[Path] = field(default_factory=list)

    # Model selection by task type
    selection_generate: SelectionConfig = field(default_factory=SelectionConfig)
    selection_embed: SelectionConfig = field(default_factory=SelectionConfig)

    # Resolved model path - set after model resolution
    path: Path | None = None

    def get(self, name: str) -> ModelConfig:
        """Get config for a model, falling back to defaults."""
        return self.models.get(name, self.defaults)

    def get_selection(self, task: str = "generate") -> SelectionConfig:
        """Get selection config for a task type."""
        if task == "embed":
            return self.selection_embed
        return self.selection_generate

    @classmethod
    def from_dict(cls, data: dict) -> "ModelsConfig":
        """Create ModelsConfig from dict (full YAML).

        Expected structure:
            locations:
              - /path/to/models
            selection:
              generate:
                path: ~/.selected.yaml
                default: qwen2.5-1.5b
              embed:
                path: ~/.selected.embed.yaml
                default: bge-small-en-v1.5
            models:
              model-name:
                task: embed
                ...
            defaults:
              system_prompt: null
              ...
        """
        # Parse per-model configs (guard against None value)
        models = {}
        for name, model_data in (data.get("models") or {}).items():
            models[name] = ModelConfig.from_dict(name, model_data)

        # Parse defaults (guard against None value)
        defaults_data = data.get("defaults") or {}
        defaults = ModelConfig.from_dict("defaults", defaults_data)

        # Parse locations - convert to Path objects
        locations_raw = data.get("locations", [])
        locations = [Path(loc) for loc in locations_raw] if locations_raw else []

        # Parse selection config (by task type, guard against None values)
        selection = data.get("selection", {}) or {}

        return cls(
            models=models,
            defaults=defaults,
            locations=locations,
            selection_generate=SelectionConfig.from_dict(
                selection.get("generate") or {}
            ),
            selection_embed=SelectionConfig.from_dict(selection.get("embed") or {}),
        )

    @classmethod
    def from_raw_config(cls, raw_config: dict[str, Any]) -> "ModelsConfig":
        """Create ModelsConfig from raw appinfra config dict.

        Args:
            raw_config: Full config dict from appinfra (includes 'models' key).

        Returns:
            ModelsConfig parsed from the models section.
        """
        models_data = raw_config.get("models") or {}
        return cls.from_dict(models_data)


def load_models_config(path: str | Path) -> ModelsConfig:
    """Load models configuration from YAML file.

    Note: Prefer using ModelsConfig.from_dict() with config data from appinfra,
    which supports !include and !path directives.

    Args:
        path: Path to models.yaml file.

    Returns:
        ModelsConfig with all model configurations.
    """
    path = Path(path)
    if not path.exists():
        return ModelsConfig()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ModelsConfig.from_dict(data)
