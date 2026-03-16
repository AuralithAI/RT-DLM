"""
Comprehensive Tests for KV Prefix Cache

Tests cover:
- Basic store/get operations
- Cache eviction policies (LRU, FIFO, LFU)
- Cache statistics and memory tracking
- Multi-layer storage
- Edge cases (capacity, invalidation, clearing)
- Integration with AttentionConfig

References:
    - vLLM: Efficient Memory Management for Large Language Model Serving
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.config.agi_config import AGIConfig
from src.core.model.advanced_attention import (
    KVPrefixCache,
    AttentionConfig,
    estimate_kv_cache_size,
)


# =============================================================================
# Basic Cache Operations
# =============================================================================

class TestKVPrefixCacheBasic:
    """Tests for basic KV Prefix Cache operations."""

    @pytest.fixture
    def cache(self):
        return KVPrefixCache(
            num_layers=4,
            max_prefix_len=32,
            num_kv_heads=4,
            head_dim=16,
            max_entries=8,
            eviction_policy="lru",
        )

    @pytest.fixture
    def sample_kv(self):
        """Generate sample key/value tensors."""
        k = jnp.ones((1, 4, 16, 16))  # [batch, kv_heads, seq, head_dim]
        v = jnp.ones((1, 4, 16, 16))
        return k, v

    def test_initial_state(self, cache):
        """Test cache is empty on creation."""
        assert cache.size == 0
        assert not cache.is_full
        assert cache.max_entries == 8

    def test_store_and_get(self, cache, sample_kv):
        """Test storing and retrieving KV pairs."""
        k, v = sample_kv
        cache.store("prefix_a", layer=0, keys=k, values=v)

        retrieved_k, retrieved_v = cache.get("prefix_a", layer=0)

        assert retrieved_k is not None
        assert retrieved_v is not None
        assert retrieved_k.shape == k.shape
        assert retrieved_v.shape == v.shape

    def test_store_converts_dtype(self, cache, sample_kv):
        """Test that stored tensors are converted to cache dtype."""
        k = jnp.ones((1, 4, 16, 16), dtype=jnp.float32)
        v = jnp.ones((1, 4, 16, 16), dtype=jnp.float32)

        cache.store("prefix_a", layer=0, keys=k, values=v)
        retrieved_k, retrieved_v = cache.get("prefix_a", layer=0)

        assert retrieved_k.dtype == jnp.bfloat16
        assert retrieved_v.dtype == jnp.bfloat16

    def test_get_nonexistent_prefix(self, cache):
        """Test getting a non-existent prefix returns None."""
        k, v = cache.get("nonexistent", layer=0)
        assert k is None
        assert v is None

    def test_get_nonexistent_layer(self, cache, sample_kv):
        """Test getting a non-existent layer returns None."""
        k, v = sample_kv
        cache.store("prefix_a", layer=0, keys=k, values=v)

        result_k, result_v = cache.get("prefix_a", layer=1)
        assert result_k is None
        assert result_v is None

    def test_has_prefix(self, cache, sample_kv):
        """Test has() method for prefix existence."""
        k, v = sample_kv
        cache.store("prefix_a", layer=0, keys=k, values=v)

        assert cache.has("prefix_a")
        assert cache.has("prefix_a", layer=0)
        assert not cache.has("prefix_a", layer=1)
        assert not cache.has("prefix_b")

    def test_multi_layer_storage(self, cache, sample_kv):
        """Test storing KV pairs across multiple layers."""
        k, v = sample_kv

        for layer in range(4):
            cache.store("prefix_a", layer=layer, keys=k * (layer + 1), values=v * (layer + 1))

        assert cache.size == 1  # Only one prefix entry

        for layer in range(4):
            rk, rv = cache.get("prefix_a", layer=layer)
            assert rk is not None
            # Check the values scale correctly
            expected = (layer + 1)
            np.testing.assert_allclose(
                float(rk.mean()), expected, atol=0.1
            )

    def test_multiple_prefixes(self, cache, sample_kv):
        """Test storing multiple different prefixes."""
        k, v = sample_kv

        cache.store("system_v1", layer=0, keys=k, values=v)
        cache.store("system_v2", layer=0, keys=k * 2, values=v * 2)
        cache.store("fewshot_3", layer=0, keys=k * 3, values=v * 3)

        assert cache.size == 3

        r1, _ = cache.get("system_v1", layer=0)
        r2, _ = cache.get("system_v2", layer=0)
        r3, _ = cache.get("fewshot_3", layer=0)

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None

    def test_overwrite_same_prefix_layer(self, cache, sample_kv):
        """Test that re-storing overwrites existing entry."""
        k, v = sample_kv

        cache.store("prefix_a", layer=0, keys=k, values=v)
        cache.store("prefix_a", layer=0, keys=k * 5, values=v * 5)

        rk, _ = cache.get("prefix_a", layer=0)
        np.testing.assert_allclose(float(rk.mean()), 5.0, atol=0.1)

    def test_prefix_len_truncation(self):
        """Test that sequences longer than max_prefix_len are truncated."""
        cache = KVPrefixCache(
            num_layers=1,
            max_prefix_len=8,
            num_kv_heads=2,
            head_dim=4,
            max_entries=4,
        )

        # Sequence length 16 > max_prefix_len 8
        k = jnp.ones((1, 2, 16, 4))
        v = jnp.ones((1, 2, 16, 4))

        cache.store("long_prefix", layer=0, keys=k, values=v)
        rk, rv = cache.get("long_prefix", layer=0)

        assert rk.shape[2] == 8  # Truncated to max_prefix_len


# =============================================================================
# Cache Eviction Tests
# =============================================================================

class TestKVPrefixCacheEviction:
    """Tests for cache eviction policies."""

    def _make_cache(self, policy, max_entries=3):
        return KVPrefixCache(
            num_layers=1,
            max_prefix_len=8,
            num_kv_heads=2,
            head_dim=4,
            max_entries=max_entries,
            eviction_policy=policy,
        )

    def _sample_kv(self):
        return jnp.ones((1, 2, 4, 4)), jnp.ones((1, 2, 4, 4))

    def test_lru_eviction(self):
        """Test LRU eviction removes least recently used."""
        cache = self._make_cache("lru", max_entries=3)
        k, v = self._sample_kv()

        cache.store("a", 0, k, v)
        cache.store("b", 0, k, v)
        cache.store("c", 0, k, v)

        # Access 'a' to make it recently used
        cache.get("a", 0)

        # Adding 'd' should evict 'b' (least recently used)
        cache.store("d", 0, k, v)

        assert cache.has("a")
        assert not cache.has("b")
        assert cache.has("c")
        assert cache.has("d")

    def test_fifo_eviction(self):
        """Test FIFO eviction removes oldest inserted."""
        cache = self._make_cache("fifo", max_entries=3)
        k, v = self._sample_kv()

        cache.store("a", 0, k, v)
        cache.store("b", 0, k, v)
        cache.store("c", 0, k, v)

        # Even if we access 'a', FIFO should still evict it first
        cache.get("a", 0)

        cache.store("d", 0, k, v)

        assert not cache.has("a")  # First in, first out
        assert cache.has("b")
        assert cache.has("c")
        assert cache.has("d")

    def test_lfu_eviction(self):
        """Test LFU eviction removes least frequently used."""
        cache = self._make_cache("lfu", max_entries=3)
        k, v = self._sample_kv()

        cache.store("a", 0, k, v)
        cache.store("b", 0, k, v)
        cache.store("c", 0, k, v)

        # Access 'a' and 'c' multiple times
        cache.get("a", 0)
        cache.get("a", 0)
        cache.get("c", 0)

        # 'b' has fewest accesses (only 1 from store), should be evicted
        cache.store("d", 0, k, v)

        assert cache.has("a")
        assert not cache.has("b")
        assert cache.has("c")
        assert cache.has("d")

    def test_eviction_chain(self):
        """Test multiple evictions maintain correctness."""
        cache = self._make_cache("lru", max_entries=2)
        k, v = self._sample_kv()

        cache.store("a", 0, k, v)
        cache.store("b", 0, k, v)
        assert cache.size == 2

        cache.store("c", 0, k, v)  # Evicts 'a'
        assert cache.size == 2
        assert not cache.has("a")

        cache.store("d", 0, k, v)  # Evicts 'b'
        assert not cache.has("b")
        assert cache.has("c")
        assert cache.has("d")

    def test_no_eviction_when_updating(self):
        """Test that updating existing prefix doesn't trigger eviction."""
        cache = self._make_cache("lru", max_entries=2)
        k, v = self._sample_kv()

        cache.store("a", 0, k, v)
        cache.store("b", 0, k, v)

        # Update 'a' - should not evict anything
        cache.store("a", 0, k * 2, v * 2)

        assert cache.size == 2
        assert cache.has("a")
        assert cache.has("b")


# =============================================================================
# Cache Management Tests
# =============================================================================

class TestKVPrefixCacheManagement:
    """Tests for cache invalidation, clearing, and stats."""

    @pytest.fixture
    def cache(self):
        return KVPrefixCache(
            num_layers=2,
            max_prefix_len=16,
            num_kv_heads=4,
            head_dim=8,
            max_entries=8,
        )

    def test_invalidate_existing(self, cache):
        """Test invalidating an existing entry."""
        k = jnp.ones((1, 4, 8, 8))
        v = jnp.ones((1, 4, 8, 8))
        cache.store("prefix_a", 0, k, v)

        assert cache.invalidate("prefix_a") is True
        assert not cache.has("prefix_a")
        assert cache.size == 0

    def test_invalidate_nonexistent(self, cache):
        """Test invalidating a non-existent entry."""
        assert cache.invalidate("nonexistent") is False

    def test_clear(self, cache):
        """Test clearing all entries."""
        k = jnp.ones((1, 4, 8, 8))
        v = jnp.ones((1, 4, 8, 8))

        for i in range(5):
            cache.store(f"prefix_{i}", 0, k, v)

        assert cache.size == 5
        cache.clear()
        assert cache.size == 0

    def test_stats(self, cache):
        """Test cache statistics."""
        k = jnp.ones((1, 4, 8, 8))
        v = jnp.ones((1, 4, 8, 8))

        cache.store("prefix_a", 0, k, v)
        cache.store("prefix_a", 1, k, v)
        cache.get("prefix_a", 0)
        cache.get("prefix_a", 0)

        stats = cache.get_stats()

        assert stats["num_entries"] == 1
        assert stats["max_entries"] == 8
        assert stats["utilization"] == pytest.approx(1 / 8)
        assert stats["total_elements"] > 0
        assert stats["memory_mb"] > 0
        assert stats["eviction_policy"] == "lru"
        assert "prefix_a" in stats["access_counts"]

    def test_is_full(self):
        """Test is_full property."""
        cache = KVPrefixCache(
            num_layers=1, max_prefix_len=4,
            num_kv_heads=1, head_dim=4, max_entries=2
        )
        k = jnp.ones((1, 1, 4, 4))
        v = jnp.ones((1, 1, 4, 4))

        assert not cache.is_full
        cache.store("a", 0, k, v)
        assert not cache.is_full
        cache.store("b", 0, k, v)
        assert cache.is_full

    def test_float32_dtype(self):
        """Test cache with float32 dtype."""
        cache = KVPrefixCache(
            num_layers=1, max_prefix_len=8,
            num_kv_heads=2, head_dim=4,
            max_entries=4, dtype=jnp.float32,
        )

        k = jnp.ones((1, 2, 4, 4))
        v = jnp.ones((1, 2, 4, 4))
        cache.store("a", 0, k, v)

        rk, rv = cache.get("a", 0)
        assert rk.dtype == jnp.float32

    def test_seq_axis_1_format(self):
        """Test with [batch, seq_len, num_kv_heads, head_dim] format."""
        cache = KVPrefixCache(
            num_layers=1, max_prefix_len=8,
            num_kv_heads=4, head_dim=8, max_entries=4,
        )

        # This format has seq as axis 1
        k = jnp.ones((1, 16, 4, 8))  # seq_len=16 > max_prefix=8
        v = jnp.ones((1, 16, 4, 8))

        cache.store("prefix_a", 0, k, v)
        rk, _ = cache.get("prefix_a", 0)

        assert rk.shape[1] == 8  # Truncated


# =============================================================================
# Integration Tests
# =============================================================================

class TestKVCacheIntegration:
    """Integration tests combining KV cache with config and attention."""

    def test_config_kv_cache_settings(self):
        """Test AGIConfig KV cache settings exist and validate."""
        config = AGIConfig(
            enable_kv_cache=True,
            kv_cache_prefix_len=128,
            kv_cache_max_batch=16,
            kv_cache_eviction="fifo",
        )

        assert config.enable_kv_cache is True
        assert config.kv_cache_prefix_len == 128
        assert config.kv_cache_max_batch == 16
        assert config.kv_cache_eviction == "fifo"

    def test_create_cache_from_config(self):
        """Test creating a KV cache from AGIConfig parameters."""
        config = AGIConfig(
            enable_kv_cache=True,
            kv_cache_prefix_len=64,
            kv_cache_max_batch=8,
            kv_cache_eviction="lru",
            d_model=128,
            num_heads=8,
        )

        num_kv_heads = getattr(config, 'num_kv_heads', None) or config.num_heads
        head_dim = config.d_model // config.num_heads

        cache = KVPrefixCache(
            num_layers=config.num_layers,
            max_prefix_len=config.kv_cache_prefix_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_entries=config.kv_cache_max_batch,
            eviction_policy=config.kv_cache_eviction,
        )

        assert cache.max_entries == 8
        assert cache.max_prefix_len == 64
        assert cache.eviction_policy == "lru"

    def test_estimate_kv_cache_size_function(self):
        """Test the KV cache size estimation utility."""
        result = estimate_kv_cache_size(
            batch_size=1,
            seq_len=2048,
            d_model=4096,
            num_layers=32,
            num_kv_heads=8,
            num_heads=32,
            dtype_bytes=2,
        )

        assert "kv_cache_gb" in result
        assert "per_token_mb" in result
        assert "elements" in result
        assert result["kv_cache_gb"] > 0
        assert result["per_token_mb"] > 0

    def test_cache_workflow(self):
        """Test a realistic cache workflow: store system prompt, reuse across requests."""
        cache = KVPrefixCache(
            num_layers=2,
            max_prefix_len=32,
            num_kv_heads=4,
            head_dim=16,
            max_entries=4,
        )

        # Simulate storing system prompt KV across layers
        for layer in range(2):
            k = jax.random.normal(jax.random.PRNGKey(layer), (1, 4, 20, 16))
            v = jax.random.normal(jax.random.PRNGKey(layer + 10), (1, 4, 20, 16))
            cache.store("system_prompt_v1", layer, k, v)

        # Simulate 3 different user requests reusing the cached prefix
        for req_id in range(3):
            for layer in range(2):
                cached_k, cached_v = cache.get("system_prompt_v1", layer)
                assert cached_k is not None

                # New tokens for this request
                new_k = jax.random.normal(
                    jax.random.PRNGKey(100 + req_id), (1, 4, 8, 16)
                )
                new_v = jax.random.normal(
                    jax.random.PRNGKey(200 + req_id), (1, 4, 8, 16)
                )

                # Concatenate cached prefix with new tokens
                full_k = jnp.concatenate([cached_k, new_k], axis=2)
                full_v = jnp.concatenate([cached_v, new_v], axis=2)

                assert full_k.shape[2] == 20 + 8  # prefix + new

        stats = cache.get_stats()
        assert stats["access_counts"]["system_prompt_v1"] >= 7  # store(2) + get(6)
