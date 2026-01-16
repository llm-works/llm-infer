"""LoRA adapter discovery and management.

Scans a directory for adapters, reads their config.yaml, and tracks enabled ones.
No database - purely file-based discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from appinfra.log import Logger


def validate_adapter_id(adapter_id: str, base_path: Path) -> Path | None:
    """Validate adapter_id and resolve to safe path within base_path.

    Performs security checks to prevent path traversal attacks:
    - Rejects path separators (/, \\)
    - Rejects parent directory references (..)
    - Ensures resolved path stays within base_path

    Note on symlinks: Both base_path and adapter directories may be symlinks.
    The validation resolves all symlinks before checking containment, so a
    symlinked adapter directory is allowed as long as its real path is within
    the real base_path. This enables sharing adapter weights across deployments
    while maintaining security boundaries.

    Args:
        adapter_id: The adapter name to validate.
        base_path: The base directory adapters must reside in.

    Returns:
        Resolved Path if valid, None if validation fails.
    """
    # Reject empty, ".", path separators, and traversal sequences
    if not adapter_id or adapter_id == ".":
        return None
    if "/" in adapter_id or "\\" in adapter_id or ".." in adapter_id:
        return None

    resolved_base = base_path.resolve()
    adapter_path = (base_path / adapter_id).resolve()

    # Ensure resolved path stays within base_path
    if not adapter_path.is_relative_to(resolved_base):
        return None

    return adapter_path


@dataclass
class LoadedAdapter:
    """An adapter that has been discovered and is enabled."""

    adapter_id: str
    path: Path
    enabled: bool = True
    description: str | None = None
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class AdapterManager:
    """Manages LoRA adapter discovery and tracking.

    Scans a base directory for adapter subdirectories, reads their config.yaml,
    and tracks which adapters are enabled for inference.

    Directory structure:
        base_path/
          adapter-name-1/
            config.yaml      # enabled: true/false, optional description
            adapter_model.safetensors
            ...
          adapter-name-2/
            config.yaml
            ...

    Config format (config.yaml):
        enabled: true
        description: "Optional description"
    """

    CONFIG_FILENAME = "config.yaml"

    def __init__(self, base_path: Path | str | None, lg: Logger | None = None) -> None:
        self._base_path = Path(base_path).expanduser() if base_path else None
        self._lg = lg
        self._adapters: dict[str, LoadedAdapter] = {}

    @property
    def base_path(self) -> Path | None:
        return self._base_path

    def _is_scannable(self) -> bool:
        """Check if base_path is valid for scanning."""
        if not self._base_path or not self._base_path.exists():
            return False
        if not self._base_path.is_dir():
            if self._lg:
                self._lg.warning(
                    "adapter base_path is not a directory",
                    extra={"path": str(self._base_path)},
                )
            return False
        return True

    def scan(self) -> int:
        """Scan base_path for adapters and load enabled ones.

        Returns:
            Number of enabled adapters found.
        """
        self._adapters.clear()
        if not self._is_scannable():
            return 0

        count = 0
        for entry in self._base_path.iterdir():  # type: ignore[union-attr]
            if not entry.is_dir():
                continue
            adapter = self._load_adapter(entry)
            if adapter and adapter.enabled:
                self._adapters[adapter.adapter_id] = adapter
                count += 1
                if self._lg:
                    self._lg.debug(
                        "adapter loaded",
                        extra={
                            "adapter_id": adapter.adapter_id,
                            "path": str(adapter.path),
                        },
                    )

        if self._lg:
            self._lg.info("adapter scan complete", extra={"enabled_count": count})
        return count

    def _load_adapter(self, path: Path) -> LoadedAdapter | None:
        """Load adapter config from a directory."""
        config_path = path / self.CONFIG_FILENAME
        if not config_path.exists():
            # No config = skip silently
            return None

        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            if self._lg:
                self._lg.warning(
                    "failed to read adapter config",
                    extra={"path": str(config_path), "error": str(e)},
                )
            return None

        if not isinstance(config, dict):
            if self._lg:
                self._lg.warning(
                    "adapter config must be a mapping",
                    extra={"path": str(config_path), "type": type(config).__name__},
                )
            return None

        return LoadedAdapter(
            adapter_id=path.name,
            path=path,
            enabled=config.get("enabled", False),
            description=config.get("description"),
        )

    def list(self) -> list[LoadedAdapter]:
        """List all enabled adapters."""
        return list(self._adapters.values())

    def get(self, adapter_id: str) -> LoadedAdapter | None:
        """Get an adapter by ID."""
        return self._adapters.get(adapter_id)

    def is_available(self, adapter_id: str) -> bool:
        """Check if an adapter is available for use."""
        return adapter_id in self._adapters

    def resolve_path(self, adapter_id: str) -> Path | None:
        """Resolve adapter_id to its full path.

        Returns None if adapter is not registered (but inference may still
        work if the path exists - this is just for validation).
        """
        adapter = self._adapters.get(adapter_id)
        return adapter.path if adapter else None

    def refresh_one(self, adapter_id: str) -> LoadedAdapter | None:
        """Refresh a single adapter by re-reading its config.

        Args:
            adapter_id: The adapter directory name to refresh.

        Returns:
            The adapter if enabled, None if disabled or not found.
        """
        if not self._base_path:
            return None

        # Validate adapter_id to prevent path traversal
        path = validate_adapter_id(adapter_id, self._base_path)
        if path is None:
            if self._lg:
                self._lg.warning(
                    "rejected adapter_id with invalid characters",
                    extra={"adapter_id": adapter_id},
                )
            return None

        if not path.exists() or not path.is_dir():
            # Remove if it was previously loaded but no longer exists
            self._adapters.pop(adapter_id, None)
            return None

        adapter = self._load_adapter(path)
        if adapter and adapter.enabled:
            self._adapters[adapter_id] = adapter
            if self._lg:
                self._lg.debug("adapter refreshed", extra={"adapter_id": adapter_id})
            return adapter
        else:
            # Disabled or invalid - remove from loaded set
            self._adapters.pop(adapter_id, None)
            if self._lg:
                self._lg.debug("adapter unloaded", extra={"adapter_id": adapter_id})
            return None
