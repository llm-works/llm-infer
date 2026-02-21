"""LoRA adapter metadata computation.

Provides reusable utilities for computing adapter fingerprints (mtime, md5)
for verification and logging purposes.

Example:
    >>> from llm_infer.adapter_meta import compute_adapter_metadata
    >>> meta = compute_adapter_metadata("/path/to/adapter")
    >>> print(f"Modified: {meta.mtime}, Hash: {meta.md5}")
    Modified: 2026-02-16T02:51:20Z, Hash: 2732c092187a
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# PEFT standard weight filenames (tried in order)
WEIGHT_FILES = ["adapter_model.safetensors", "adapter_model.bin"]


@dataclass(frozen=True)
class AdapterMetadata:
    """Immutable metadata for a LoRA adapter.

    Attributes:
        mtime: ISO-8601 formatted modification time of the weights file.
        md5: First 12 characters of MD5 hash of the weights file.
        weights_file: Name of the weights file used (for debugging).
    """

    mtime: str
    md5: str
    weights_file: str | None = None

    @classmethod
    def unknown(cls) -> AdapterMetadata:
        """Return metadata indicating weights file not found."""
        return cls(mtime="unknown", md5="unknown", weights_file=None)

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for logging extras."""
        return {"mtime": self.mtime, "md5": self.md5}


def compute_adapter_metadata(adapter_path: Path | str) -> AdapterMetadata:
    """Compute metadata for a LoRA adapter.

    Finds the weights file (safetensors or bin format) and computes:
    - mtime: ISO-8601 formatted modification timestamp
    - md5: First 12 characters of MD5 hash (sufficient for identification)

    Args:
        adapter_path: Path to the adapter directory containing weights.

    Returns:
        AdapterMetadata with computed values, or unknown values if
        weights file not found.
    """
    adapter_path = Path(adapter_path)

    # Find weights file
    weights_file: Path | None = None
    for filename in WEIGHT_FILES:
        candidate = adapter_path / filename
        if candidate.exists():
            weights_file = candidate
            break

    if weights_file is None:
        return AdapterMetadata.unknown()

    # Compute mtime
    mtime = datetime.fromtimestamp(weights_file.stat().st_mtime, tz=UTC)
    mtime_str = mtime.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Compute md5 (streaming to handle large files)
    md5 = hashlib.md5()
    with open(weights_file, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    return AdapterMetadata(
        mtime=mtime_str,
        md5=md5.hexdigest()[:12],
        weights_file=weights_file.name,
    )
