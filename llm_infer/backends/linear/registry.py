"""Backend registry with priority-based auto-selection.

This module provides a registry for quantized linear backends, allowing:
- Registration of backends with priorities
- Auto-selection of best available backend per format
- Manual backend selection by name
- Graceful fallback when preferred backend unavailable
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .formats.base import QuantFormat, QuantizedLinearBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Registry: format -> [(backend_class, priority)]
# Higher priority backends are preferred when available
_BACKEND_REGISTRY: dict[QuantFormat, list[tuple[type, int]]] = {
    QuantFormat.AWQ: [],
    QuantFormat.FP8: [],
}


def register_backend(
    format: QuantFormat,
    backend_cls: type,
    priority: int = 0,
) -> None:
    """Register a backend for a quantization format.

    Args:
        format: Quantization format this backend supports
        backend_cls: Backend class (must implement QuantizedLinearBackend protocol)
        priority: Selection priority (higher = preferred). Use:
            0: Fallback (always available, e.g., PyTorch)
            100: Optimized (hardware-specific, e.g., Marlin/CUTLASS)
    """
    if format not in _BACKEND_REGISTRY:
        _BACKEND_REGISTRY[format] = []

    _BACKEND_REGISTRY[format].append((backend_cls, priority))
    # Keep sorted by priority descending (highest first)
    _BACKEND_REGISTRY[format].sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        f"Registered backend {backend_cls.__name__} for {format.name} "
        f"with priority {priority}"
    )


def _try_preferred_backend(
    backends: list, preference: str, format: QuantFormat
) -> QuantizedLinearBackend | None:
    """Try to find the preferred backend. Returns None if not available."""
    for backend_cls, _ in backends:
        backend: QuantizedLinearBackend = backend_cls()
        if backend.name == preference and backend.is_available():
            logger.debug(f"Using preferred backend: {backend.name}")
            return backend
    logger.warning(
        f"Preferred backend '{preference}' not available for {format.name}, falling back to auto-selection"
    )
    return None


def get_backend(
    format: QuantFormat, preference: str | None = None
) -> QuantizedLinearBackend:
    """Get the best available backend for a quantization format."""
    if format not in _BACKEND_REGISTRY or not _BACKEND_REGISTRY[format]:
        raise ValueError(f"No backends registered for format {format.name}")

    backends = _BACKEND_REGISTRY[format]
    if preference:
        preferred = _try_preferred_backend(backends, preference, format)
        if preferred:
            return preferred

    for backend_cls, priority in backends:
        candidate: QuantizedLinearBackend = backend_cls()
        if candidate.is_available():
            logger.debug(
                f"Auto-selected backend {candidate.name} (priority {priority}) for {format.name}"
            )
            return candidate

    raise RuntimeError(
        f"No backends available for format {format.name}. Registered: {[cls.__name__ for cls, _ in backends]}"
    )


def get_available_backends(format: QuantFormat) -> list[str]:
    """Get names of all available backends for a format.

    Args:
        format: Quantization format to check

    Returns:
        List of available backend names, sorted by priority (best first)
    """
    if format not in _BACKEND_REGISTRY:
        return []

    available = []
    for backend_cls, _ in _BACKEND_REGISTRY[format]:
        backend = backend_cls()
        if backend.is_available():
            available.append(backend.name)

    return available


def _auto_register_backends() -> None:
    """Auto-register all known backends with appropriate priorities.

    Called at module import time to populate the registry.
    """
    # AWQ kernels
    try:
        from .kernels.awq_pytorch import PyTorchAWQBackend

        register_backend(QuantFormat.AWQ, PyTorchAWQBackend, priority=0)
    except ImportError:
        pass

    try:
        from .kernels.awq_marlin import MarlinAWQBackend

        register_backend(QuantFormat.AWQ, MarlinAWQBackend, priority=100)
    except ImportError:
        pass

    # FP8 kernels
    try:
        from .kernels.fp8_pytorch import PyTorchFP8Backend

        register_backend(QuantFormat.FP8, PyTorchFP8Backend, priority=0)
    except ImportError:
        pass

    try:
        from .kernels.fp8_cutlass import CutlassFP8Backend

        register_backend(QuantFormat.FP8, CutlassFP8Backend, priority=100)
    except ImportError:
        pass


# Auto-register on import
_auto_register_backends()


# Backward compatibility alias for existing code
def get_linear_backend(backend_name: str = "auto") -> QuantizedLinearBackend:
    """Get an AWQ linear backend (backward compatibility).

    This is a compatibility shim for existing code that uses get_linear_backend().

    Args:
        backend_name: Backend name ("pytorch", "marlin", or "auto")

    Returns:
        AWQ backend instance
    """
    preference = None if backend_name == "auto" else backend_name
    return get_backend(QuantFormat.AWQ, preference=preference)
