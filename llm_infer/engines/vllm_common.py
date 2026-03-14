"""Common utilities for vLLM-based engines.

Shared functionality between vllm (Python API) and vllm-server (HTTP API) engines.
"""

from appinfra.log import Logger


def get_gpu_total_memory_gb() -> float | None:
    """Get total GPU memory in GB, or None if unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        total_bytes: int = torch.cuda.get_device_properties(device).total_memory
        return float(total_bytes) / (1024**3)
    except Exception:
        return None


def resolve_gpu_memory_utilization(
    lg: Logger,
    gpu_memory_gb: float | None,
    gpu_memory_utilization: float,
) -> float:
    """Convert gpu_memory_gb to utilization fraction if set.

    Args:
        lg: Logger for info/warning messages.
        gpu_memory_gb: Absolute GB limit, or None to use utilization directly.
        gpu_memory_utilization: Fraction of GPU memory (0-1), used if gpu_memory_gb is None.

    Returns:
        GPU memory utilization fraction (0.01 to 0.95).
    """
    if gpu_memory_gb is None:
        return gpu_memory_utilization

    if gpu_memory_gb <= 0:
        raise ValueError(f"gpu_memory_gb must be positive, got {gpu_memory_gb}")

    total_gb = get_gpu_total_memory_gb()
    if total_gb is None:
        lg.warning("gpu_memory_gb set but GPU detection failed, using utilization")
        return gpu_memory_utilization

    # Convert GB to fraction, capped at 0.95 (vLLM recommendation)
    utilization = min(gpu_memory_gb / total_gb, 0.95)
    utilization = max(utilization, 0.01)  # At least 1%

    lg.info(
        "gpu_memory_gb converted to utilization",
        extra={
            "gpu_memory_gb": gpu_memory_gb,
            "total_gb": round(total_gb, 1),
            "utilization": round(utilization, 4),
        },
    )
    return utilization
