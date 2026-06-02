"""Logging helpers shared across the package."""

from __future__ import annotations


def fmt_error(error: object, limit: int | None = 200) -> str:
    """Collapse whitespace and truncate for single-line logs.

    Log aggregators treat newlines as record boundaries; collapse first so
    the character budget goes to signal, not indentation.
    """
    s = " ".join(str(error).split())
    return s[:limit] if limit is not None else s
