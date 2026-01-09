"""Attention mechanisms: RoPE, KV operations, and backends."""

from typing import TYPE_CHECKING

from torch import Tensor

from ..protocols import AttentionBackend, KVCacheStorage
from .backends import FLASHINFER_AVAILABLE, FlashInferBackend, NaiveAttentionBackend
from .kv_ops import update_kv_cache
from .rope import apply_rope, precompute_rope_freqs, rotate_half

if TYPE_CHECKING:
    from ..kv_cache import SequenceKVCache


def get_attention_backend(preference: str = "auto") -> AttentionBackend:
    """Get an attention backend instance.

    Args:
        preference: Backend selection preference.
            - "auto": Use FlashInfer if available, else Naive (default)
            - "flashinfer": Use FlashInfer (raises if unavailable)
            - "naive": Use naive PyTorch implementation

    Returns:
        AttentionBackend instance.

    Raises:
        ValueError: If preference is not recognized.
        RuntimeError: If requested backend is not available.
    """
    if preference == "flashinfer":
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError(
                "FlashInfer backend requested but not available. "
                "Install with: pip install flashinfer"
            )
        return FlashInferBackend()
    elif preference == "naive":
        return NaiveAttentionBackend()
    elif preference == "auto":
        if FLASHINFER_AVAILABLE:
            return FlashInferBackend()
        return NaiveAttentionBackend()
    else:
        raise ValueError(
            f"Unknown attention backend: {preference}. "
            f"Valid options: auto, flashinfer, naive"
        )


def paged_attention(
    q: Tensor,
    layer_idx: int,
    kv_caches: list["SequenceKVCache"],
    kv_storage: KVCacheStorage,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Compute attention using paged KV cache.

    This is a convenience function that uses the default backend.
    For more control, use get_attention_backend() directly.
    """
    import math

    backend = get_attention_backend()
    sm_scale = 1.0 / math.sqrt(head_dim)
    return backend.forward(
        q, layer_idx, kv_caches, kv_storage, num_heads, num_kv_heads, sm_scale
    )


__all__ = [
    # Protocols
    "AttentionBackend",
    # Backends
    "FlashInferBackend",
    "NaiveAttentionBackend",
    "FLASHINFER_AVAILABLE",
    "get_attention_backend",
    # Legacy dispatcher
    "paged_attention",
    # RoPE
    "precompute_rope_freqs",
    "rotate_half",
    "apply_rope",
    # KV operations
    "update_kv_cache",
]
