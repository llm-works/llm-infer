"""Block pool for paged KV cache."""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class BlockPool:
    """
    Pre-allocated GPU memory blocks for KV cache.

    Manages a pool of fixed-size blocks that can be allocated to sequences.
    Compatible with FlashInfer's paged attention interface.
    """

    num_blocks: int
    block_size: int  # tokens per block
    num_layers: int
    num_kv_heads: int
    head_dim: int
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Cache tensors: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    k_cache: Tensor = field(init=False)
    v_cache: Tensor = field(init=False)
    free_blocks: list[int] = field(init=False)

    def __post_init__(self) -> None:
        """Allocate cache tensors on GPU."""
        shape = (
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )
        self.k_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.free_blocks = list(range(self.num_blocks))

    @property
    def num_free_blocks(self) -> int:
        """Number of available blocks."""
        return len(self.free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        """Number of blocks in use."""
        return self.num_blocks - len(self.free_blocks)

    def allocate(self) -> int:
        """
        Allocate a free block.

        Returns:
            Block index.

        Raises:
            RuntimeError: If no free blocks available.
        """
        if not self.free_blocks:
            raise RuntimeError("No free blocks available")
        return self.free_blocks.pop()

    def free(self, block_id: int) -> None:
        """Return a block to the pool."""
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid block_id: {block_id}")
        if block_id in self.free_blocks:
            raise ValueError(f"Block {block_id} is already free")
        self.free_blocks.append(block_id)

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate the requested number of blocks."""
        return len(self.free_blocks) >= num_blocks

    def memory_usage_bytes(self) -> int:
        """Total GPU memory used by cache tensors."""
        return int(self.k_cache.numel() * self.k_cache.element_size() * 2)
