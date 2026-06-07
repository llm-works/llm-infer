"""Helper functions for fallback client.

Extracted to keep fallback.py focused on client logic.
"""

from __future__ import annotations

from collections.abc import Mapping

from appinfra.log import Logger


def detect_cycles(fallbacks: Mapping[str, str], lg: Logger) -> set[str]:
    """Detect cycles in fallback pairs and log warnings.

    Args:
        fallbacks: Model fallback pairs (model -> fallback_model).
        lg: Logger for warnings.

    Returns:
        Set of models that are part of cycles.
    """
    cycle_models: set[str] = set()

    for start in fallbacks:
        if start in cycle_models:
            continue

        path: list[str] = []
        visited: set[str] = set()
        current = start

        while current in fallbacks:
            if current in visited:
                cycle_path = _build_cycle_path(start, fallbacks)
                lg.warning(
                    "cycle detected in fallback config",
                    extra={"cycle": cycle_path},
                )
                # Only add actual cycle members (from where current reappears)
                cycle_start_idx = path.index(current)
                cycle_models.update(path[cycle_start_idx:])
                break
            visited.add(current)
            path.append(current)
            current = fallbacks[current]

    return cycle_models


def _build_cycle_path(start: str, fallbacks: Mapping[str, str]) -> str:
    """Build a string representation of the cycle for logging."""
    path = [start]
    current = start
    seen: set[str] = {start}

    while current in fallbacks:
        next_model = fallbacks[current]
        path.append(next_model)
        if next_model in seen:
            break
        seen.add(next_model)
        current = next_model

    return " -> ".join(path)
