"""Naive PyTorch attention backend."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ...protocols import KVCacheStorage

if TYPE_CHECKING:
    from ...kv_cache import SequenceKVCache


def _gather_kv(
    kv_cache: "SequenceKVCache",
    kv_storage: KVCacheStorage,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[Tensor, Tensor] | None:
    """Gather K/V tensors from storage for a single sequence."""
    k_list = [kv_storage.k_cache[layer_idx, bid] for bid in kv_cache.block_ids]
    v_list = [kv_storage.v_cache[layer_idx, bid] for bid in kv_cache.block_ids]
    if not k_list:
        return None

    k = torch.cat(k_list, dim=0)[: kv_cache.num_tokens]
    v = torch.cat(v_list, dim=0)[: kv_cache.num_tokens]

    # Expand KV heads for GQA
    if num_kv_heads != num_heads:
        kv_len, head_dim = k.shape[0], k.shape[2]
        repeats = num_heads // num_kv_heads
        k = (
            k.unsqueeze(2)
            .expand(-1, -1, repeats, -1)
            .reshape(kv_len, num_heads, head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(-1, -1, repeats, -1)
            .reshape(kv_len, num_heads, head_dim)
        )
    return k, v


def _apply_causal_mask(
    scores: Tensor, seq_len: int, kv_len: int, device: str
) -> Tensor:
    """Apply causal mask for prefill (seq_len > 1)."""
    query_positions = torch.arange(seq_len, device=device)
    kv_positions = torch.arange(kv_len, device=device)
    offset = kv_len - seq_len
    mask = kv_positions.unsqueeze(0) > (query_positions.unsqueeze(1) + offset)
    return scores.masked_fill(mask.unsqueeze(1), float("-inf"))


class NaiveAttentionBackend:
    """Naive PyTorch attention implementation.

    Fallback when optimized kernels (FlashInfer, FlashAttention) are unavailable.
    """

    @property
    def name(self) -> str:
        return "naive"

    def forward(
        self,
        q: Tensor,
        layer_idx: int,
        kv_caches: list["SequenceKVCache"],
        kv_storage: KVCacheStorage,
        num_heads: int,
        num_kv_heads: int,
        sm_scale: float,
    ) -> Tensor:
        """Compute attention output using naive PyTorch implementation.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            layer_idx: Current layer index
            kv_caches: Per-sequence KV cache handles
            kv_storage: Underlying KV tensor storage
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA)
            sm_scale: Softmax scaling factor (typically 1/sqrt(head_dim))

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        batch, seq_len, _, head_dim = q.shape
        device = q.device
        dtype = q.dtype
        outputs = []

        for batch_idx, kv_cache in enumerate(kv_caches):
            kv = _gather_kv(kv_cache, kv_storage, layer_idx, num_heads, num_kv_heads)
            if kv is None:
                outputs.append(
                    torch.zeros(
                        seq_len, num_heads, head_dim, device=device, dtype=dtype
                    )
                )
                continue

            k, v = kv
            q_f, k_f, v_f = q[batch_idx].float(), k.float(), v.float()
            scores = torch.einsum("shd,khd->shk", q_f, k_f) * sm_scale

            if seq_len > 1:
                scores = _apply_causal_mask(scores, seq_len, k.shape[0], str(device))

            attn = F.softmax(scores, dim=-1)
            outputs.append(torch.einsum("shk,khd->shd", attn, v_f).to(dtype))

        return torch.stack(outputs, dim=0)
