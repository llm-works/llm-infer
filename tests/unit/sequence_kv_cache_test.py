"""Unit tests for SequenceKVCache."""

import pytest

from llm_infer.primitives.kv_cache.pool import BlockPool
from llm_infer.primitives.kv_cache.sequence import SequenceKVCache

pytestmark = pytest.mark.unit


def make_pool(num_blocks: int = 10, block_size: int = 4) -> BlockPool:
    """Create a test BlockPool on CPU."""
    return BlockPool(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        device="cpu",
    )


class TestSequenceKVCacheInit:
    """Test SequenceKVCache initialization."""

    def test_empty_on_init(self) -> None:
        """Test that cache starts empty."""
        cache = SequenceKVCache()
        assert cache.num_tokens == 0
        assert cache.num_blocks() == 0
        assert cache.block_ids == []

    def test_get_block_table_empty(self) -> None:
        """Test that empty cache returns empty block table."""
        cache = SequenceKVCache()
        assert cache.get_block_table() == []


class TestSequenceKVCacheAppend:
    """Test appending tokens to cache."""

    def test_append_first_token_allocates_block(self) -> None:
        """Test that first token allocates a block."""
        pool = make_pool(block_size=4)
        cache = SequenceKVCache()

        block_id, offset = cache.append_token(pool)

        assert cache.num_tokens == 1
        assert cache.num_blocks() == 1
        assert offset == 0

    def test_append_fills_block(self) -> None:
        """Test that tokens fill block before allocating new one."""
        pool = make_pool(block_size=4)
        cache = SequenceKVCache()

        # Fill first block (4 tokens)
        for i in range(4):
            block_id, offset = cache.append_token(pool)
            assert offset == i
            assert cache.num_blocks() == 1

        # 5th token should allocate new block
        block_id, offset = cache.append_token(pool)
        assert offset == 0
        assert cache.num_blocks() == 2

    def test_append_returns_correct_block_id(self) -> None:
        """Test that append returns correct block ID."""
        pool = make_pool(num_blocks=10, block_size=2)
        cache = SequenceKVCache()

        # First two tokens in first block
        block1, _ = cache.append_token(pool)
        block1_again, _ = cache.append_token(pool)
        assert block1 == block1_again

        # Third token in second block
        block2, _ = cache.append_token(pool)
        assert block2 != block1


class TestSequenceKVCacheAllocatePrompt:
    """Test allocating blocks for prompt."""

    def test_allocate_single_block(self) -> None:
        """Test allocating prompt that fits in one block."""
        pool = make_pool(block_size=16)
        cache = SequenceKVCache()

        cache.allocate_for_prompt(pool, num_tokens=10)

        assert cache.num_tokens == 10
        assert cache.num_blocks() == 1

    def test_allocate_multiple_blocks(self) -> None:
        """Test allocating prompt that needs multiple blocks."""
        pool = make_pool(block_size=4)
        cache = SequenceKVCache()

        cache.allocate_for_prompt(pool, num_tokens=10)

        assert cache.num_tokens == 10
        assert cache.num_blocks() == 3  # ceil(10/4) = 3

    def test_allocate_exact_block_boundary(self) -> None:
        """Test allocating prompt exactly at block boundary."""
        pool = make_pool(block_size=4)
        cache = SequenceKVCache()

        cache.allocate_for_prompt(pool, num_tokens=8)

        assert cache.num_tokens == 8
        assert cache.num_blocks() == 2


class TestSequenceKVCacheFree:
    """Test freeing cache blocks."""

    def test_free_returns_blocks_to_pool(self) -> None:
        """Test that free_all returns blocks to pool."""
        pool = make_pool(num_blocks=10, block_size=4)
        cache = SequenceKVCache()

        cache.allocate_for_prompt(pool, num_tokens=10)
        assert pool.num_free_blocks == 7  # 10 - 3 used

        cache.free_all(pool)

        assert pool.num_free_blocks == 10
        assert cache.num_tokens == 0
        assert cache.num_blocks() == 0

    def test_free_empty_cache(self) -> None:
        """Test that freeing empty cache is safe."""
        pool = make_pool()
        cache = SequenceKVCache()

        # Should not raise
        cache.free_all(pool)

        assert cache.num_tokens == 0


class TestSequenceKVCacheBlockTable:
    """Test block table retrieval."""

    def test_get_block_table_returns_copy(self) -> None:
        """Test that get_block_table returns a copy."""
        pool = make_pool(block_size=4)
        cache = SequenceKVCache()

        cache.allocate_for_prompt(pool, num_tokens=5)
        table = cache.get_block_table()

        # Modify returned table
        table.append(999)

        # Original should be unchanged
        assert 999 not in cache.block_ids

    def test_get_block_table_order(self) -> None:
        """Test that block table preserves allocation order."""
        pool = make_pool(num_blocks=10, block_size=2)
        cache = SequenceKVCache()

        # Allocate 3 blocks
        cache.allocate_for_prompt(pool, num_tokens=5)

        table = cache.get_block_table()
        assert len(table) == 3
        # Blocks should be in order they were allocated
        assert table == cache.block_ids
