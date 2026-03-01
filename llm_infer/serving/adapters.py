"""LoRA adapter discovery and management.

Scans a directory for adapters, reads their config.yaml, and tracks enabled ones.
No database - purely file-based discovery.

Versioned Adapter Resolution:
    Supports versioned adapter symlinks in the format `{base_key}-{md5}` where md5 is
    12 hex characters. Multiple versions of the same adapter can coexist, and requests
    for a base_key resolve to the latest version (by mtime).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from appinfra.log import Logger

from ..adapter_meta import compute_adapter_metadata

# Pattern for versioned adapter keys: base_key-md5 where md5 is 12 hex chars
_VERSIONED_KEY_PATTERN = re.compile(r"^(.+)-([0-9a-f]{12})$")


def parse_adapter_key(key: str) -> tuple[str, str | None]:
    """Parse adapter key to extract name and optional md5 suffix.

    Versioned keys have the format: {name}-{md5} where md5 is 12 hex characters.
    If the key doesn't match this pattern, it's treated as a name with no version.

    Args:
        key: The full adapter key (directory/symlink name).

    Returns:
        Tuple of (name, md5). md5 is None if no valid version suffix found.

    Examples:
        >>> parse_adapter_key("my-adapter-a1b2c3d4e5f6")
        ("my-adapter", "a1b2c3d4e5f6")
        >>> parse_adapter_key("my-adapter")
        ("my-adapter", None)
        >>> parse_adapter_key("simple")
        ("simple", None)
    """
    match = _VERSIONED_KEY_PATTERN.match(key)
    if match:
        return match.group(1), match.group(2)
    return key, None


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
        key: Full lookup identifier (directory name with version suffix).
        name: Logical adapter name without version suffix.
            For versioned adapters like "my-adapter-a1b2c3d4e5f6", name is
            "my-adapter". For unversioned adapters, name equals key.
        path: Filesystem path to the adapter directory.
        md5: First 12 chars of MD5 hash of weights file (for verification).
        mtime: ISO-8601 modification time of weights file.
        enabled: Whether the adapter is enabled for inference.
        description: Optional human-readable description from config.yaml.
        loaded_at: Timestamp when the adapter was loaded/refreshed.
    """

    key: str
    name: str
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
        enabled: false  # Optional, defaults to true (auto-enabled)
        description: "Optional description"
    """

    CONFIG_FILENAME = "config.yaml"
    ADAPTER_CONFIG_FILENAME = "adapter_config.json"

    def __init__(
        self,
        lg: Logger,
        base_path: Path | str | None,
        base_model_path: Path | str | None = None,
    ) -> None:
        self._lg = lg
        self._base_path = Path(base_path).expanduser() if base_path else None
        self._base_model_path = Path(base_model_path) if base_model_path else None
        self._adapters: dict[str, LoadedAdapter] = {}  # full_key → adapter
        self._versions: dict[str, list[str]] = {}  # name → [full_keys by mtime desc]

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

        Builds both the primary index (full_key → adapter) and the versions
        index (name → [full_keys sorted by mtime desc]). The versions index
        enables name-based resolution to the latest version.

        Returns:
            Number of enabled adapters found.
        """
        self._adapters.clear()
        self._versions.clear()
        if not self._is_scannable():
            return 0

        adapters_list = self._scan_and_load_adapters()
        self._populate_indexes(adapters_list)

        self._lg.info(
            "adapter scan complete",
            extra={
                "enabled_count": len(adapters_list),
                "unique_names": len(self._versions),
            },
        )
        return len(adapters_list)

    def _scan_and_load_adapters(self) -> list[LoadedAdapter]:
        """Scan directory and load enabled adapters."""
        adapters: list[LoadedAdapter] = []
        for entry in self._base_path.iterdir():  # type: ignore[union-attr]
            if not entry.is_dir():
                continue
            adapter = self._load_adapter(entry)
            if adapter and adapter.enabled:
                adapters.append(adapter)
                self._lg.debug(
                    "adapter loaded",
                    extra={
                        "key": adapter.key,
                        "adapter_name": adapter.name,
                        "path": str(adapter.path),
                        "md5": adapter.md5,
                        "mtime": adapter.mtime,
                    },
                )
        return adapters

    def _populate_indexes(self, adapters: list[LoadedAdapter]) -> None:
        """Populate primary and versions indexes from adapter list."""
        for adapter in adapters:
            self._adapters[adapter.key] = adapter
        self._build_versions_index(adapters)

    def _build_versions_index(self, adapters: list[LoadedAdapter]) -> None:
        """Build the versions index mapping name to sorted full_keys."""
        # Group by name
        by_name: dict[str, list[LoadedAdapter]] = {}
        for adapter in adapters:
            by_name.setdefault(adapter.name, []).append(adapter)

        # Sort each group by mtime descending (newest first), None values last
        for name, group in by_name.items():
            sorted_group = sorted(
                group,
                key=lambda a: a.mtime or "",  # None sorts before any string
                reverse=True,
            )
            self._versions[name] = [a.key for a in sorted_group]

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

    def _read_adapter_config_json(self, adapter_path: Path, key: str) -> str | None:
        """Read base_model_name_or_path from adapter_config.json.

        Returns the base model path string, or None if unavailable.
        """
        config_path = adapter_path / self.ADAPTER_CONFIG_FILENAME
        if not config_path.exists():
            return None

        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            base_model: str | None = config.get("base_model_name_or_path")
            return base_model
        except Exception as e:
            self._lg.warning(
                "failed to read adapter_config.json",
                extra={"key": key, "path": str(config_path), "exception": e},
            )
            return None

    def _check_base_model_compatibility(self, adapter_path: Path, key: str) -> bool:
        """Check if adapter's base model matches the current model.

        Returns True if compatible, False if incompatible.
        """
        if self._base_model_path is None:
            return True

        adapter_base_model = self._read_adapter_config_json(adapter_path, key)
        if not adapter_base_model:
            return True  # No base model info, allow

        current_model_name = self._base_model_path.name.lower()
        adapter_model_name = Path(adapter_base_model).name.lower()

        if current_model_name != adapter_model_name:
            self._lg.info(
                "adapter base model mismatch - skipping",
                extra={
                    "key": key,
                    "adapter_base_model": adapter_model_name,
                    "current_model": current_model_name,
                },
            )
            return False

        return True

    def _load_adapter(self, path: Path, key: str | None = None) -> LoadedAdapter | None:
        """Load adapter config and compute metadata from a directory.

        Args:
            path: Path to the adapter directory (may be resolved symlink target).
            key: Override key to use instead of path.name. Use when path is a
                resolved symlink but you want the original symlink name as key.
        """
        config = self._read_config(path / self.CONFIG_FILENAME)
        if config is None:
            return None

        full_key = key if key is not None else path.name

        # Check base model compatibility before loading
        if not self._check_base_model_compatibility(path, full_key):
            return None

        name, _ = parse_adapter_key(full_key)
        metadata = compute_adapter_metadata(path)
        return LoadedAdapter(
            key=full_key,
            name=name,
            path=path,
            md5=metadata.md5 if metadata.md5 != "unknown" else None,
            mtime=metadata.mtime if metadata.mtime != "unknown" else None,
            enabled=config.get("enabled", True),
            description=config.get("description"),
        )

    def list(self) -> list[LoadedAdapter]:
        """List all enabled adapters."""
        return list(self._adapters.values())

    def get(self, key: str) -> LoadedAdapter | None:
        """Get an adapter by exact full key."""
        return self._adapters.get(key)

    def resolve(self, key: str) -> LoadedAdapter | None:
        """Resolve adapter key to LoadedAdapter with version fallback.

        Resolution order:
        1. Exact match: If key matches a full adapter key, return it.
        2. Name match: If key matches an adapter name, return the latest version
           (highest mtime).

        Args:
            key: Adapter key to resolve. Can be either a full versioned key
                (e.g., "my-adapter-a1b2c3d4e5f6") or a name
                (e.g., "my-adapter").

        Returns:
            LoadedAdapter if found, None otherwise.

        Examples:
            >>> mgr.resolve("my-adapter-a1b2c3d4e5f6")  # Exact match
            LoadedAdapter(key="my-adapter-a1b2c3d4e5f6", ...)
            >>> mgr.resolve("my-adapter")  # Name → latest version
            LoadedAdapter(key="my-adapter-a1b2c3d4e5f6", ...)
        """
        # 1. Exact match (full key including version suffix)
        if key in self._adapters:
            return self._adapters[key]

        # 2. Name match (resolve to latest version by mtime)
        if key in self._versions:
            latest_key = self._versions[key][0]  # First = highest mtime
            return self._adapters[latest_key]

        return None

    def is_available(self, key: str) -> bool:
        """Check if an adapter is available for use.

        Supports both full keys and names (resolves to latest version).
        """
        return self.resolve(key) is not None

    def resolve_path(self, key: str) -> Path | None:
        """Resolve adapter key to its full path.

        Supports both full keys and names (resolves to latest version).

        Returns None if adapter is not registered (but inference may still
        work if the path exists - this is just for validation).
        """
        adapter = self.resolve(key)
        return adapter.path if adapter else None

    def _validate_refresh_path(self, key: str) -> Path | None:
        """Validate key and return path for refresh, or None if invalid."""
        if not self._base_path:
            return None
        path = validate_adapter_key(key, self._base_path)
        if path is None:
            self._lg.warning(
                "rejected adapter key with invalid characters", extra={"key": key}
            )
        return path

    def refresh_one(self, key: str) -> LoadedAdapter | None:
        """Refresh a single adapter by re-reading its config and metadata.

        After refreshing, rebuilds the versions index to maintain consistency.
        """
        path = self._validate_refresh_path(key)
        if path is None or not path.exists() or not path.is_dir():
            self._adapters.pop(key, None)
            self._rebuild_versions_index()
            return None

        # Pass original key to handle symlinked adapters correctly
        adapter = self._load_adapter(path, key=key)
        if adapter and adapter.enabled:
            self._adapters[key] = adapter
            self._rebuild_versions_index()
            self._lg.debug(
                "adapter refreshed",
                extra={"key": key, "md5": adapter.md5, "mtime": adapter.mtime},
            )
            return adapter

        # Disabled or invalid - remove from loaded set
        self._adapters.pop(key, None)
        self._rebuild_versions_index()
        self._lg.debug("adapter unloaded", extra={"key": key})
        return None

    def _rebuild_versions_index(self) -> None:
        """Rebuild versions index from current adapters."""
        self._build_versions_index(list(self._adapters.values()))
