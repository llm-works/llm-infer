"""Unit tests for KV cache pool."""

import pytest
import torch

from llm_infer.engines.native.kv_cache.pool import BlockPool

pytestmark = pytest.mark.unit


class TestBlockPoolInit:
    """Test BlockPool initialization."""

    def test_creates_cache_tensors(self) -> None:
        """Test that cache tensors are created with correct shape."""
        pool = BlockPool(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        expected_shape = (2, 10, 16, 4, 64)
        assert pool.k_cache.shape == expected_shape
        assert pool.v_cache.shape == expected_shape

    def test_all_blocks_initially_free(self) -> None:
        """Test that all blocks are free after init."""
        pool = BlockPool(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        assert pool.num_free_blocks == 10
        assert pool.num_allocated_blocks == 0

    def test_dtype_is_respected(self) -> None:
        """Test that specified dtype is used."""
        pool = BlockPool(
            num_blocks=4,
            block_size=8,
            num_layers=1,
            num_kv_heads=2,
            head_dim=32,
            device="cpu",
            dtype=torch.float32,
        )

        assert pool.k_cache.dtype == torch.float32
        assert pool.v_cache.dtype == torch.float32


class TestBlockPoolAllocation:
    """Test block allocation and freeing."""

    def test_allocate_returns_block_id(self) -> None:
        """Test that allocate returns a valid block id."""
        pool = BlockPool(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        block_id = pool.allocate()
        assert 0 <= block_id < 10

    def test_allocate_decreases_free_count(self) -> None:
        """Test that allocate decreases free block count."""
        pool = BlockPool(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        pool.allocate()
        assert pool.num_free_blocks == 9
        assert pool.num_allocated_blocks == 1

    def test_allocate_multiple_blocks(self) -> None:
        """Test allocating multiple blocks."""
        pool = BlockPool(
            num_blocks=5,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        blocks = [pool.allocate() for _ in range(5)]
        assert len(set(blocks)) == 5  # All unique
        assert pool.num_free_blocks == 0

    def test_allocate_raises_when_empty(self) -> None:
        """Test that allocate raises when no blocks available."""
        pool = BlockPool(
            num_blocks=2,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        pool.allocate()
        pool.allocate()

        with pytest.raises(RuntimeError, match="No free blocks"):
            pool.allocate()

    def test_free_returns_block_to_pool(self) -> None:
        """Test that free returns block to pool."""
        pool = BlockPool(
            num_blocks=5,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        block_id = pool.allocate()
        assert pool.num_free_blocks == 4

        pool.free(block_id)
        assert pool.num_free_blocks == 5

    def test_free_invalid_block_raises(self) -> None:
        """Test that freeing invalid block raises."""
        pool = BlockPool(
            num_blocks=5,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        with pytest.raises(ValueError, match="Invalid block_id"):
            pool.free(10)

        with pytest.raises(ValueError, match="Invalid block_id"):
            pool.free(-1)

    def test_free_already_free_block_raises(self) -> None:
        """Test that freeing already free block raises."""
        pool = BlockPool(
            num_blocks=5,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        with pytest.raises(ValueError, match="already free"):
            pool.free(0)


class TestBlockPoolQueries:
    """Test pool query methods."""

    def test_can_allocate_true(self) -> None:
        """Test can_allocate returns True when enough blocks."""
        pool = BlockPool(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        assert pool.can_allocate(5) is True
        assert pool.can_allocate(10) is True

    def test_can_allocate_false(self) -> None:
        """Test can_allocate returns False when not enough blocks."""
        pool = BlockPool(
            num_blocks=5,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
        )

        assert pool.can_allocate(6) is False

    def test_memory_usage_bytes(self) -> None:
        """Test memory_usage_bytes calculation."""
        pool = BlockPool(
            num_blocks=10,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
            dtype=torch.float16,
        )

        # k_cache: 2 * 10 * 16 * 4 * 64 = 81920 elements
        # element_size for float16 = 2 bytes
        # k + v = 2 caches
        expected = 81920 * 2 * 2
        assert pool.memory_usage_bytes() == expected
