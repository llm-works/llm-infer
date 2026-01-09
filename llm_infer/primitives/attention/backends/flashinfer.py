"""FlashInfer attention backend."""

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from ...protocols import KVCacheStorage

if TYPE_CHECKING:
    from ...kv_cache import SequenceKVCache

try:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    BatchDecodeWithPagedKVCacheWrapper = None  # type: ignore[assignment, misc]
    BatchPrefillWithPagedKVCacheWrapper = None  # type: ignore[assignment, misc]


# Workspace buffer size (128MB as recommended by FlashInfer docs)
_WORKSPACE_SIZE = 128 * 1024 * 1024


class FlashInferBackend:
    """FlashInfer optimized attention implementation.

    Uses FlashInfer's paged attention kernels for efficient GPU execution.
    Supports both decode (single token) and prefill (multiple tokens) phases.
    """

    def __init__(self) -> None:
        self._workspace_buffer: Tensor | None = None
        self._decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None
        self._prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None

    @property
    def name(self) -> str:
        return "flashinfer"

    @staticmethod
    def is_available() -> bool:
        """Check if FlashInfer is available."""
        return FLASHINFER_AVAILABLE

    def _ensure_workspace(self, device: torch.device) -> Tensor:
        """Lazily allocate workspace buffer on the correct device."""
        if self._workspace_buffer is None or self._workspace_buffer.device != device:
            self._workspace_buffer = torch.zeros(
                _WORKSPACE_SIZE, dtype=torch.uint8, device=device
            )
            # Reset wrappers when device changes
            self._decode_wrapper = None
            self._prefill_wrapper = None
        return self._workspace_buffer

    def _get_decode_wrapper(
        self, device: torch.device
    ) -> "BatchDecodeWithPagedKVCacheWrapper":
        """Get or create decode wrapper."""
        workspace = self._ensure_workspace(device)
        if self._decode_wrapper is None:
            self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD"
            )
        return self._decode_wrapper

    def _get_prefill_wrapper(
        self, device: torch.device
    ) -> "BatchPrefillWithPagedKVCacheWrapper":
        """Get or create prefill wrapper."""
        workspace = self._ensure_workspace(device)
        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace, kv_layout="NHD"
            )
        return self._prefill_wrapper

    def _build_paging_indices(
        self,
        kv_caches: list["SequenceKVCache"],
        kv_storage: KVCacheStorage,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build paging index tensors for FlashInfer."""
        indptr_list: list[int] = [0]
        indices_list: list[int] = []
        last_page_lens_list: list[int] = []

        for kv_cache in kv_caches:
            indices_list.extend(kv_cache.block_ids)
            indptr_list.append(len(indices_list))
            last_page_len = kv_cache.num_tokens % kv_storage.block_size
            last_page_lens_list.append(
                last_page_len if last_page_len > 0 else kv_storage.block_size
            )

        kv_indptr = torch.tensor(indptr_list, dtype=torch.int32, device=device)
        kv_indices = torch.tensor(indices_list, dtype=torch.int32, device=device)
        kv_last_page_lens = torch.tensor(
            last_page_lens_list, dtype=torch.int32, device=device
        )
        return kv_indptr, kv_indices, kv_last_page_lens

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
        """Compute attention output using FlashInfer."""
        if not FLASHINFER_AVAILABLE:
            raise RuntimeError("FlashInfer is not available")

        _, seq_len, _, head_dim = q.shape
        kv_indptr, kv_indices, kv_last_page_lens = self._build_paging_indices(
            kv_caches, kv_storage, q.device
        )
        paged_kv_cache = (kv_storage.k_cache[layer_idx], kv_storage.v_cache[layer_idx])
        args = (
            q,
            paged_kv_cache,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_storage.block_size,
            sm_scale,
        )

        if seq_len == 1:
            return self._decode_forward(*args)
        return self._prefill_forward(*args[:5], kv_caches, *args[5:])

    def _decode_forward(
        self,
        q: Tensor,
        paged_kv_cache: tuple[Tensor, Tensor],
        kv_indptr: Tensor,
        kv_indices: Tensor,
        kv_last_page_lens: Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        sm_scale: float,
    ) -> Tensor:
        """Decode attention for single token per sequence."""
        wrapper = self._get_decode_wrapper(q.device)

        # Plan the attention computation
        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",  # We apply RoPE separately
            q_data_type=q.dtype,
            sm_scale=sm_scale,
        )

        # q shape: [batch, 1, num_heads, head_dim] -> [batch, num_heads, head_dim]
        q_squeezed = q.squeeze(1)

        # Run attention
        output = wrapper.run(q_squeezed, paged_kv_cache)

        # output shape: [batch, num_heads, head_dim] -> [batch, 1, num_heads, head_dim]
        return output.unsqueeze(1)

    def _plan_prefill(
        self,
        wrapper: Any,
        q: Tensor,
        kv_indptr: Tensor,
        kv_indices: Tensor,
        kv_last_page_lens: Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        sm_scale: float,
    ) -> Tensor:
        """Plan prefill attention and return query indptr."""
        batch_size, seq_len = q.shape[:2]
        qo_indptr = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=q.device
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            pos_encoding_mode="NONE",
            q_data_type=q.dtype,
            sm_scale=sm_scale,
        )
        return qo_indptr

    def _prefill_forward(
        self,
        q: Tensor,
        paged_kv_cache: tuple[Tensor, Tensor],
        kv_indptr: Tensor,
        kv_indices: Tensor,
        kv_last_page_lens: Tensor,
        kv_caches: list["SequenceKVCache"],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        sm_scale: float,
    ) -> Tensor:
        """Prefill attention for multiple tokens per sequence."""
        wrapper = self._get_prefill_wrapper(q.device)
        batch_size, seq_len = q.shape[:2]

        self._plan_prefill(
            wrapper,
            q,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
            sm_scale,
        )

        output = wrapper.run(q.reshape(-1, num_heads, head_dim), paged_kv_cache)
        return output.reshape(batch_size, seq_len, num_heads, head_dim)
