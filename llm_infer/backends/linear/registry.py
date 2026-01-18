"""Backend resolution with priority-based auto-selection.

This module provides a BackendRegistry class to resolve quantized linear backends:
- Auto-selection of best available backend per format
- Manual backend selection by name
- Graceful fallback when preferred backend unavailable

No global state - create a BackendRegistry instance where needed.
"""

from __future__ import annotations

import importlib
from typing import Any

from appinfra.log import Logger

from .formats.base import QuantFormat, QuantizedLinearBackend

# Backend candidates per format: (module, class_name, name, priority)
# Sorted by priority descending (highest first)
_AWQ_BACKENDS = [
    ("awq_marlin", "MarlinAWQBackend", "marlin", 100),
    ("awq_pytorch", "PyTorchAWQBackend", "pytorch", 0),
]

_FP8_BACKENDS = [
    ("fp8_cutlass", "CutlassFP8Backend", "cutlass", 100),
    ("fp8_pytorch", "PyTorchFP8Backend", "pytorch", 0),
]

_BACKENDS_BY_FORMAT: dict[QuantFormat, list[tuple[str, str, str, int]]] = {
    QuantFormat.AWQ: _AWQ_BACKENDS,
    QuantFormat.FP8: _FP8_BACKENDS,
}


class BackendRegistry:
    """Registry for resolving quantized linear backends.

    Backends are resolved on demand by trying imports in priority order.

    Example:
        >>> registry = BackendRegistry(lg)
        >>> backend = registry.get(QuantFormat.AWQ)
        >>> backend = registry.get(QuantFormat.AWQ, preference="pytorch")
    """

    def __init__(self, lg: Logger) -> None:
        """Initialize the registry.

        Args:
            lg: Logger instance for debug/warning messages.
        """
        self._lg = lg

    def _try_import(self, module_name: str, class_name: str) -> type[Any] | None:
        """Try to import a backend class. Returns None if import fails."""
        try:
            module = importlib.import_module(f".kernels.{module_name}", __package__)
            return getattr(module, class_name)  # type: ignore[no-any-return]
        except ImportError:
            return None

    def _try_preferred(
        self,
        candidates: list[tuple[str, str, str, int]],
        preference: str,
        format: QuantFormat,
    ) -> QuantizedLinearBackend | None:
        """Try to get the preferred backend. Returns None if not available."""
        for module_name, class_name, name, _ in candidates:
            if name != preference:
                continue
            backend_cls = self._try_import(module_name, class_name)
            if backend_cls:
                backend: QuantizedLinearBackend = backend_cls(self._lg)
                if backend.is_available():
                    self._lg.debug("using preferred backend", extra={"backend": name})
                    return backend
            self._lg.warning(
                "preferred backend not available, falling back",
                extra={"preference": preference, "format": format.name},
            )
            break
        return None

    def _try_auto_select(
        self, candidates: list[tuple[str, str, str, int]], format: QuantFormat
    ) -> QuantizedLinearBackend | None:
        """Try backends in priority order, return first available."""
        for module_name, class_name, name, priority in candidates:
            backend_cls = self._try_import(module_name, class_name)
            if backend_cls is None:
                continue
            backend: QuantizedLinearBackend = backend_cls(self._lg)
            if backend.is_available():
                self._lg.debug(
                    "auto-selected backend",
                    extra={
                        "backend": name,
                        "priority": priority,
                        "format": format.name,
                    },
                )
                return backend
        return None

    def get(
        self, format: QuantFormat, preference: str | None = None
    ) -> QuantizedLinearBackend:
        """Get the best available backend for a quantization format.

        Args:
            format: Quantization format to get backend for.
            preference: Optional backend name preference (e.g., "marlin", "pytorch").

        Returns:
            Backend instance implementing QuantizedLinearBackend protocol.

        Raises:
            ValueError: If format is not supported.
            RuntimeError: If no backends are available for the format.
        """
        candidates = _BACKENDS_BY_FORMAT.get(format)
        if candidates is None:
            raise ValueError(f"No backends defined for format {format.name}")

        if preference:
            backend = self._try_preferred(candidates, preference, format)
            if backend:
                return backend

        backend = self._try_auto_select(candidates, format)
        if backend:
            return backend

        raise RuntimeError(
            f"No backends available for format {format.name}. "
            f"Tried: {[name for _, _, name, _ in candidates]}"
        )

    def list_available(self, format: QuantFormat) -> list[str]:
        """Get names of all available backends for a format.

        Args:
            format: Quantization format to check.

        Returns:
            List of available backend names, sorted by priority (best first).
        """
        candidates = _BACKENDS_BY_FORMAT.get(format)
        if candidates is None:
            return []

        available = []
        for module_name, class_name, name, _ in candidates:
            backend_cls = self._try_import(module_name, class_name)
            if backend_cls is None:
                continue
            backend: QuantizedLinearBackend = backend_cls(self._lg)
            if backend.is_available():
                available.append(name)

        return available
