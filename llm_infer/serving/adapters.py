"""LoRA adapter discovery and management.

Scans a directory for adapters, reads their config.yaml, and tracks enabled ones.
No database - purely file-based discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from appinfra.log import Logger

from ..adapter_meta import compute_adapter_metadata


def validate_adapter_key(adapter_key: str, base_path: Path) -> Path | None:
    """Validate adapter key and resolve to safe path.

    Performs security checks to prevent path traversal attacks:
    - Rejects path separators (/, \\)
    - Rejects parent directory references (..)
    - Ensures logical path stays within base_path (before symlink resolution)

    Symlinks are supported: adapter directories may be symlinks pointing
    outside base_path (e.g., to a shared registry). Containment is checked
    before resolving symlinks, so the user input is validated against the
    logical path structure. Symlink targets are operator-controlled and trusted.

    Args:
        adapter_key: The adapter key to validate.
        base_path: The base directory containing adapter entries.

    Returns:
        Resolved Path if valid, None if validation fails.
    """
    # Reject empty, ".", path separators, and traversal sequences
    if not adapter_key or adapter_key == ".":
        return None
    if "/" in adapter_key or "\\" in adapter_key or ".." in adapter_key:
        return None

    # Check containment BEFORE resolving symlinks (validates user input)
    adapter_path_logical = base_path / adapter_key
    if not adapter_path_logical.is_relative_to(base_path):
        return None

    # Resolve symlinks for actual use (symlink targets are operator-controlled)
    return adapter_path_logical.resolve()


@dataclass
class LoadedAdapter:
    """An adapter that has been discovered and is enabled.

    Attributes:
        key: Lookup identifier (directory name, or structured id in future).
        path: Filesystem path to the adapter directory.
        md5: First 12 chars of MD5 hash of weights file (for verification).
        mtime: ISO-8601 modification time of weights file.
        enabled: Whether the adapter is enabled for inference.
        description: Optional human-readable description from config.yaml.
        loaded_at: Timestamp when the adapter was loaded/refreshed.
    """

    key: str
    path: Path
    md5: str | None = None
    mtime: str | None = None
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

    def __init__(self, lg: Logger, base_path: Path | str | None) -> None:
        self._lg = lg
        self._base_path = Path(base_path).expanduser() if base_path else None
        self._adapters: dict[str, LoadedAdapter] = {}

    @property
    def base_path(self) -> Path | None:
        return self._base_path

    def _is_scannable(self) -> bool:
        """Check if base_path is valid for scanning."""
        if not self._base_path or not self._base_path.exists():
            return False
        if not self._base_path.is_dir():
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
                self._adapters[adapter.key] = adapter
                count += 1
                self._lg.debug(
                    "adapter loaded",
                    extra={
                        "key": adapter.key,
                        "path": str(adapter.path),
                        "md5": adapter.md5,
                        "mtime": adapter.mtime,
                    },
                )

        self._lg.info("adapter scan complete", extra={"enabled_count": count})
        return count

    def _read_config(self, config_path: Path) -> dict | None:
        """Read and validate adapter config.yaml file."""
        if not config_path.exists():
            return None

        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            self._lg.warning(
                "failed to read adapter config",
                extra={"path": str(config_path), "exception": e},
            )
            return None

        if not isinstance(config, dict):
            self._lg.warning(
                "adapter config must be a mapping",
                extra={"path": str(config_path), "type": type(config).__name__},
            )
            return None

        return config

    def _load_adapter(self, path: Path) -> LoadedAdapter | None:
        """Load adapter config and compute metadata from a directory."""
        config = self._read_config(path / self.CONFIG_FILENAME)
        if config is None:
            return None

        metadata = compute_adapter_metadata(path)
        return LoadedAdapter(
            key=path.name,
            path=path,
            md5=metadata.md5 if metadata.md5 != "unknown" else None,
            mtime=metadata.mtime if metadata.mtime != "unknown" else None,
            enabled=config.get("enabled", False),
            description=config.get("description"),
        )

    def list(self) -> list[LoadedAdapter]:
        """List all enabled adapters."""
        return list(self._adapters.values())

    def get(self, key: str) -> LoadedAdapter | None:
        """Get an adapter by key."""
        return self._adapters.get(key)

    def is_available(self, key: str) -> bool:
        """Check if an adapter is available for use."""
        return key in self._adapters

    def resolve_path(self, key: str) -> Path | None:
        """Resolve adapter key to its full path.

        Returns None if adapter is not registered (but inference may still
        work if the path exists - this is just for validation).
        """
        adapter = self._adapters.get(key)
        return adapter.path if adapter else None

    def refresh_one(self, key: str) -> LoadedAdapter | None:
        """Refresh a single adapter by re-reading its config and metadata.

        Args:
            key: The adapter key (directory name) to refresh.

        Returns:
            The adapter if enabled, None if disabled or not found.
        """
        if not self._base_path:
            return None

        # Validate key to prevent path traversal
        path = validate_adapter_key(key, self._base_path)
        if path is None:
            self._lg.warning(
                "rejected adapter key with invalid characters",
                extra={"key": key},
            )
            return None

        if not path.exists() or not path.is_dir():
            # Remove if it was previously loaded but no longer exists
            self._adapters.pop(key, None)
            return None

        adapter = self._load_adapter(path)
        if adapter and adapter.enabled:
            self._adapters[key] = adapter
            self._lg.debug(
                "adapter refreshed",
                extra={"key": key, "md5": adapter.md5, "mtime": adapter.mtime},
            )
            return adapter
        else:
            # Disabled or invalid - remove from loaded set
            self._adapters.pop(key, None)
            self._lg.debug("adapter unloaded", extra={"key": key})
            return None
