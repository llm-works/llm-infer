"""Backwards compatibility re-exports.

Model configuration has moved to llm_infer.models.
This module re-exports for backwards compatibility.
"""

from pathlib import Path

from ...models import (
    ModelConfig,
    ModelsConfig,
    SelectionConfig,
    ThinkConfig,
    load_models_config,
)


# Legacy function - kept for backwards compatibility
def get_selected_model_name(selection_path: str | Path | None = None) -> str | None:
    """Get the currently selected model name from selection file.

    Deprecated: Use ModelResolver.load_selection_file() instead.
    """
    import yaml

    if selection_path is None:
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


__all__ = [
    "ModelsConfig",
    "ModelConfig",
    "SelectionConfig",
    "ThinkConfig",
    "load_models_config",
    "get_selected_model_name",
]
