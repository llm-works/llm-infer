"""Model resolution logic.

Resolves model paths from names, selection files, and configured locations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from logging import Logger

    from .config import ModelsConfig, SelectionConfig


class ModelResolver:
    """Resolves model paths from names, selection files, and locations.

    Resolution priority:
    1. Direct path (if provided)
    2. Model name lookup in locations
    3. Selection file (ops-controlled selected.yaml)
    4. Default model name from config
    """

    def __init__(self, locations: list[Path], lg: Logger | None = None) -> None:
        """Initialize resolver with search locations.

        Args:
            locations: Directories to search for models, in priority order.
            lg: Optional logger for debug/error output.
        """
        self.locations = locations
        self.lg = lg

    def find_by_name(self, name: str) -> Path | None:
        """Find model by name in configured locations.

        Searches each location for a subdirectory matching the name
        that contains a config.json file (indicating a valid model).

        Args:
            name: Model name (subdirectory name).

        Returns:
            Path to model directory if found, None otherwise.
        """
        for loc in self.locations:
            model_path = loc / name
            if model_path.is_dir() and (model_path / "config.json").exists():
                return model_path
        return None

    def load_selection_file(self, path: str | Path) -> tuple[str | None, Path | None]:
        """Load model selection from YAML file.

        Selection files contain either:
        - name: model name to look up in locations
        - path: direct path to model directory

        Args:
            path: Path to selection YAML file.

        Returns:
            Tuple of (model_name, model_path). One or both may be None.
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if data is None:
                return None, None
            model_name = data.get("name")
            model_path = data.get("path")
            return model_name, Path(model_path) if model_path else None
        except FileNotFoundError:
            if self.lg:
                self.lg.debug("selection file not found", extra={"path": str(path)})
            return None, None
        except Exception as e:
            if self.lg:
                self.lg.warning(
                    "failed to load selection file",
                    extra={"path": str(path), "error": str(e)},
                )
            return None, None

    def resolve(
        self,
        model_path: Path | None = None,
        model_name: str | None = None,
        selection: SelectionConfig | None = None,
    ) -> Path | None:
        """Resolve model path using priority chain.

        Resolution order:
        1. Direct model_path if provided
        2. model_name lookup in locations
        3. Selection file path/name
        4. Selection default name
        """
        if model_path:
            return self._resolve_direct_path(model_path)
        if model_name:
            return self._resolve_by_name(model_name)
        if selection:
            path = self._resolve_from_selection(selection)
            if path:
                return path
        if self.lg:
            self.lg.error("no model specified")
        return None

    def _resolve_direct_path(self, model_path: Path) -> Path | None:
        """Validate and return direct model path."""
        if not model_path.exists():
            if self.lg:
                self.lg.error(
                    "model path does not exist", extra={"path": str(model_path)}
                )
            return None
        return model_path

    def _resolve_by_name(self, model_name: str) -> Path | None:
        """Look up model by name in configured locations."""
        path = self.find_by_name(model_name)
        if path:
            return path
        if self.lg:
            self.lg.error(
                "model not found",
                extra={
                    "model": model_name,
                    "locations": [str(p) for p in self.locations],
                },
            )
        return None

    def _resolve_from_selection(self, selection: SelectionConfig) -> Path | None:
        """Resolve model from selection config."""
        # Try selection file first
        if selection.path:
            sel_name, sel_path = self.load_selection_file(selection.path)
            if sel_path:
                return self._try_selection_path(sel_path, selection.path)
            if sel_name:
                return self._try_selection_name(sel_name, selection.path)

        # Fall back to default
        if selection.default:
            return self._try_default_model(selection.default)

        return None

    def _try_selection_path(self, sel_path: Path, selection_file: str) -> Path | None:
        """Try to use direct path from selection file."""
        if self.lg:
            self.lg.debug(
                "using selection file path",
                extra={"path": str(sel_path), "file": selection_file},
            )
        if not sel_path.exists():
            if self.lg:
                self.lg.error(
                    "selection model_path does not exist", extra={"path": str(sel_path)}
                )
            return None
        return sel_path

    def _try_selection_name(self, sel_name: str, selection_file: str) -> Path | None:
        """Try to find model by name from selection file."""
        if self.lg:
            self.lg.debug(
                "using selection file name",
                extra={"name": sel_name, "file": selection_file},
            )
        path = self.find_by_name(sel_name)
        if path:
            return path
        if self.lg:
            self.lg.error(
                "selection model not found",
                extra={
                    "model": sel_name,
                    "locations": [str(p) for p in self.locations],
                },
            )
        return None

    def _try_default_model(self, default_name: str) -> Path | None:
        """Try to find the default model by name."""
        path = self.find_by_name(default_name)
        if path:
            return path
        if self.lg:
            self.lg.error(
                "default model not found",
                extra={
                    "model": default_name,
                    "locations": [str(p) for p in self.locations],
                },
            )
        return None


def create_resolver(config: ModelsConfig, lg: Logger | None = None) -> ModelResolver:
    """Create a ModelResolver from ModelsConfig.

    Args:
        config: Models configuration with locations.
        lg: Optional logger.

    Returns:
        Configured ModelResolver.
    """
    return ModelResolver(config.locations, lg)
