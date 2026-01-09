"""KV cache operations."""

from typing import TYPE_CHECKING

from torch import Tensor

from ..protocols import KVCacheStorage

if TYPE_CHECKING:
    from ..kv_cache import SequenceKVCache


def update_kv_cache(
    k: Tensor,
    v: Tensor,
    layer_idx: int,
    kv_caches: list["SequenceKVCache"],
    kv_storage: KVCacheStorage,
) -> None:
    """Store K, V tensors in the paged cache."""
    batch, seq_len, _, _ = k.shape

    for batch_idx, kv_cache in enumerate(kv_caches):
        for token_idx in range(seq_len):
            token_pos = kv_cache.num_tokens - seq_len + token_idx
            block_idx = token_pos // kv_storage.block_size
            offset = token_pos % kv_storage.block_size

            if block_idx < len(kv_cache.block_ids):
                block_id = kv_cache.block_ids[block_idx]
                kv_storage.k_cache[layer_idx, block_id, offset] = k[
                    batch_idx, token_idx
                ]
                kv_storage.v_cache[layer_idx, block_id, offset] = v[
                    batch_idx, token_idx
                ]
