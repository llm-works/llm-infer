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

        visited: set[str] = set()
        current = start

        while current in fallbacks:
            if current in visited:
                cycle_path = _build_cycle_path(start, fallbacks)
                lg.warning(
                    "cycle detected in fallback config",
                    extra={"cycle": cycle_path},
                )
                cycle_models.update(visited)
                break
            visited.add(current)
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


def build_model_chain(
    model: str | None,
    fallbacks: Mapping[str, str],
) -> list[str | None]:
    """Build list of models to try by following fallback pairs.

    Args:
        model: Starting model (or None for default).
        fallbacks: Model fallback pairs.

    Returns:
        Ordered list of models to try.
    """
    if model is None:
        return [None]

    chain: list[str | None] = [model]
    seen = {model}
    current = model

    while current in fallbacks:
        next_model = fallbacks[current]
        if next_model in seen:
            break
        chain.append(next_model)
        seen.add(next_model)
        current = next_model

    return chain
