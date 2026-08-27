# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-infer Authors

"""Helper functions for fallback client.

Extracted to keep fallback.py focused on client logic.
"""

from __future__ import annotations

from collections.abc import Mapping

from appinfra.log import Logger


def parse_fallback_key(key: str) -> tuple[str, str | None]:
    """Split a fallback map key or value into ``(model, backend_or_none)``.

    The fallback map accepts two forms:

    * ``"model"`` — bare, backend is resolved from the router's routing table
      (unambiguous only when a single backend serves the model).
    * ``"model@backend"`` — qualified, pins the model to the named backend.
      Chosen over ``/`` because ``/`` collides with OpenRouter's own
      ``provider/model`` convention.

    Args:
        key: A fallback map key or value.

    Returns:
        ``(model, backend)`` where ``backend`` is ``None`` for bare keys.

    Raises:
        ValueError: If ``@`` appears with either side empty.

    Examples:
        >>> parse_fallback_key("gpt-4o")
        ('gpt-4o', None)
        >>> parse_fallback_key("gpt-4o@openai")
        ('gpt-4o', 'openai')
    """
    if "@" not in key:
        return key, None
    model, _, backend = key.partition("@")
    if not model or not backend:
        raise ValueError(
            f"invalid fallback ref {key!r}: both model and backend are required "
            "(expected 'model@backend')"
        )
    return model, backend


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
