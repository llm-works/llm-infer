"""Model-specific configuration loader."""

from dataclasses import dataclass, field
from pathlib import Path

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
    system_prompt: str | None = None
    think: ThinkConfig = field(default_factory=ThinkConfig)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "ModelConfig":
        """Create ModelConfig from dict (YAML section)."""
        think_data = data.get("think", {})
        return cls(
            name=name,
            system_prompt=data.get("system_prompt"),
            think=ThinkConfig.from_dict(think_data),
        )


@dataclass
class ModelsConfig:
    """Container for all model configurations."""

    models: dict[str, ModelConfig] = field(default_factory=dict)
    defaults: ModelConfig = field(default_factory=lambda: ModelConfig(name="defaults"))

    def get(self, name: str) -> ModelConfig:
        """Get config for a model, falling back to defaults."""
        return self.models.get(name, self.defaults)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelsConfig":
        """Create ModelsConfig from dict (full YAML)."""
        models = {}
        for name, model_data in data.get("models", {}).items():
            models[name] = ModelConfig.from_dict(name, model_data)

        defaults_data = data.get("defaults", {})
        defaults = ModelConfig.from_dict("defaults", defaults_data)

        return cls(models=models, defaults=defaults)


def load_models_config(path: str | Path) -> ModelsConfig:
    """Load models configuration from YAML file.

    Args:
        path: Path to models.yaml file.

    Returns:
        ModelsConfig with all model configurations.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    path = Path(path)
    if not path.exists():
        # Return empty config with defaults if file doesn't exist
        return ModelsConfig()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return ModelsConfig.from_dict(data)


def get_selected_model_name(selection_path: str | Path | None = None) -> str | None:
    """Get the currently selected model name from selection file.

    Args:
        selection_path: Path to selected.yaml. Defaults to ~/.models/selected.yaml
            or ~/ops/models/selected.yaml.

    Returns:
        Model name or None if not found.
    """
    if selection_path is None:
        # Try common locations
        candidates = [
            Path.home() / "ops" / "models" / "selected.yaml",
            Path.home() / ".models" / "selected.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                selection_path = candidate
                break

    if selection_path is None:
        return None

    path = Path(selection_path)
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data.get("name")
    except Exception:
        return None
