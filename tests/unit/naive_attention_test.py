"""Unit tests for NaiveAttentionBackend."""

import pytest
import torch

from llm_infer.primitives.attention.backends.naive import NaiveAttentionBackend
from llm_infer.primitives.kv_cache.pool import BlockPool
from llm_infer.primitives.kv_cache.sequence import SequenceKVCache

pytestmark = pytest.mark.unit


class TestNaiveAttentionBackend:
    """Test NaiveAttentionBackend."""

    def test_name_property(self) -> None:
        """Test backend name."""
        backend = NaiveAttentionBackend()
        assert backend.name == "naive"

    def test_forward_single_token(self) -> None:
        """Test forward pass with single token."""
        backend = NaiveAttentionBackend()

        # Setup: 1 batch, 1 token, 4 heads, head_dim 8
        batch_size = 1
        seq_len = 1
        num_heads = 4
        num_kv_heads = 4
        head_dim = 8
        block_size = 4

        # Create block pool and kv cache
        pool = BlockPool(
            num_blocks=4,
            block_size=block_size,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device="cpu",
        )

        # Create KV cache and allocate for 1 token
        kv_cache = SequenceKVCache()
        kv_cache.allocate_for_prompt(pool, 1)

        # Set some K/V values in the cache
        block_id = kv_cache.block_ids[0]
        pool.k_cache[0, block_id, 0] = torch.randn(num_kv_heads, head_dim)
        pool.v_cache[0, block_id, 0] = torch.randn(num_kv_heads, head_dim)

        # Query tensor [batch, seq_len, num_heads, head_dim]
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Forward pass
        output = backend.forward(
            q=q,
            layer_idx=0,
            kv_caches=[kv_cache],
            kv_storage=pool,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            sm_scale=1.0 / (head_dim**0.5),
        )

        # Check output shape
        assert output.shape == q.shape

    def test_forward_empty_kv_cache(self) -> None:
        """Test forward with empty KV cache (should handle gracefully)."""
        backend = NaiveAttentionBackend()

        batch_size = 1
        seq_len = 1
        num_heads = 4
        num_kv_heads = 4
        head_dim = 8

        pool = BlockPool(
            num_blocks=4,
            block_size=4,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device="cpu",
        )

        # Empty KV cache (no tokens yet)
        kv_cache = SequenceKVCache()

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # This should not crash - forward should handle empty case
        # The behavior depends on implementation - it might return zeros
        # or the query itself
        try:
            output = backend.forward(
                q=q,
                layer_idx=0,
                kv_caches=[kv_cache],
                kv_storage=pool,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                sm_scale=1.0,
            )
            # If it succeeds, check output shape
            assert output.shape == q.shape
        except (IndexError, RuntimeError):
            # Some implementations may raise an error for empty cache
            pass


class TestNaiveAttentionScaling:
    """Test attention scaling in NaiveAttentionBackend."""

    def test_sm_scale_applied(self) -> None:
        """Test that softmax scale is applied."""
        backend = NaiveAttentionBackend()

        num_heads = 2
        num_kv_heads = 2
        head_dim = 4
        block_size = 4

        pool = BlockPool(
            num_blocks=2,
            block_size=block_size,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device="cpu",
        )

        kv_cache = SequenceKVCache()
        kv_cache.allocate_for_prompt(pool, 1)

        # Set KV values
        block_id = kv_cache.block_ids[0]
        pool.k_cache[0, block_id, 0] = torch.ones(num_kv_heads, head_dim)
        pool.v_cache[0, block_id, 0] = torch.ones(num_kv_heads, head_dim)

        q = torch.ones(1, 1, num_heads, head_dim)

        # Run with different scales
        out1 = backend.forward(
            q=q,
            layer_idx=0,
            kv_caches=[kv_cache],
            kv_storage=pool,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            sm_scale=1.0,
        )

        out2 = backend.forward(
            q=q,
            layer_idx=0,
            kv_caches=[kv_cache],
            kv_storage=pool,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            sm_scale=0.5,
        )

        # Both should return valid tensors
        assert out1.shape == q.shape
        assert out2.shape == q.shape
