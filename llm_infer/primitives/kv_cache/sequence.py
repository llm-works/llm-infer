"""Per-sequence KV cache management."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pool import BlockPool


@dataclass
class SequenceKVCache:
    """
    Page table for one sequence's KV cache.

    Tracks which blocks are allocated to this sequence and current token count.
    """

    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0

    def num_blocks(self) -> int:
        """Number of blocks allocated to this sequence."""
        return len(self.block_ids)

    def append_token(self, pool: "BlockPool") -> tuple[int, int]:
        """
        Allocate space for the next token.

        Args:
            pool: Block pool to allocate from.

        Returns:
            Tuple of (block_id, offset within block).
        """
        offset = self.num_tokens % pool.block_size
        if offset == 0:
            # Need a new block
            block_id = pool.allocate()
            self.block_ids.append(block_id)
        self.num_tokens += 1
        return self.block_ids[-1], (self.num_tokens - 1) % pool.block_size

    def allocate_for_prompt(self, pool: "BlockPool", num_tokens: int) -> None:
        """
        Allocate blocks for a prompt of given length.

        Args:
            pool: Block pool to allocate from.
            num_tokens: Number of tokens in the prompt.
        """
        num_blocks_needed = (num_tokens + pool.block_size - 1) // pool.block_size
        for _ in range(num_blocks_needed):
            block_id = pool.allocate()
            self.block_ids.append(block_id)
        self.num_tokens = num_tokens

    def free_all(self, pool: "BlockPool") -> None:
        """Return all blocks to the pool."""
        for block_id in self.block_ids:
            pool.free(block_id)
        self.block_ids.clear()
        self.num_tokens = 0

    def get_block_table(self) -> list[int]:
        """Get the block table for FlashInfer."""
        return self.block_ids.copy()
