"""
Advanced Attention Mechanisms for RT-DLM AGI

This module implements state-of-the-art attention variants to improve efficiency,
scalability, and performance beyond standard Multi-Head Attention (MHA).

Key Components:
- RoPE (Rotary Position Embedding): Better long-context handling and extrapolation
- GQA (Grouped-Query Attention): 2-4x KV cache reduction with minimal quality loss
- MQA (Multi-Query Attention): Extreme efficiency with shared KV across all heads
- FlashAttention: Memory-efficient O(n) attention via tiling (when available)
- LinearAttention: Approximate attention with O(n) complexity
- SlidingWindowAttention: Local attention for very long sequences

Design Philosophy:
- Backward compatible with existing SelfAttentionModel
- Configurable via agi_config.py flags
- Integrates with spiking/pruning mechanisms
- Works with tensor parallelism in scalable_training.py

References:
- RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
- GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)
- FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)
- Longformer: The Long-Document Transformer (Beltagy et al., 2020)
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for advanced attention mechanisms."""

    d_model: int = 384
    num_heads: int = 8
    num_kv_heads: Optional[int] = None  # For GQA/MQA (None = standard MHA)
    max_seq_length: int = 2048
    dropout_rate: float = 0.1

    # Position encoding
    position_encoding: Literal["learned", "rope", "alibi", "none"] = "rope"
    rope_theta: float = 10000.0  # RoPE base frequency
    rope_scaling: Optional[float] = None  # For extended context (e.g., 2.0 for 2x length)

    # Attention variant
    attention_type: Literal["standard", "gqa", "mqa", "linear", "sliding"] = "standard"
    sliding_window_size: int = 512  # For sliding window attention

    # Efficiency options
    use_flash_attention: bool = False  # Use Flash Attention if available
    use_memory_efficient: bool = True  # Use memory-efficient computation
    causal: bool = True  # Causal masking for autoregressive

    # Spiking integration
    enable_spiking: bool = True
    spike_threshold: float = 0.1


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================


def precompute_rope_frequencies(
    dim: int, max_seq_length: int, theta: float = 10000.0, scaling_factor: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute rotary embedding frequencies.

    Args:
        dim: Dimension per head (must be even)
        max_seq_length: Maximum sequence length
        theta: Base frequency (default 10000.0 from RoFormer)
        scaling_factor: Optional scaling for extended context

    Returns:
        cos_cached: Cosine frequencies [max_seq_length, dim//2]
        sin_cached: Sine frequencies [max_seq_length, dim//2]
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    # Apply scaling for extended context (NTK-aware scaling)
    if scaling_factor is not None and scaling_factor > 1.0:
        # NTK-aware interpolation for better extrapolation
        inv_freq = inv_freq / (scaling_factor ** (dim / (dim - 2)))

    # Compute positions
    positions = jnp.arange(max_seq_length, dtype=jnp.float32)

    # Outer product: [max_seq_length, dim//2]
    freqs = jnp.outer(positions, inv_freq)

    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(
    x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, position_ids: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Apply Rotary Position Embedding to input tensor.

    Args:
        x: Input tensor [..., seq_len, dim]
        cos: Precomputed cosines [max_seq_len, dim//2]
        sin: Precomputed sines [max_seq_len, dim//2]
        position_ids: Optional position indices [batch, seq_len]

    Returns:
        Rotated tensor with same shape as x
    """
    seq_len = x.shape[-2]
    dim = x.shape[-1]

    # Get relevant positions
    if position_ids is not None:
        cos = cos[position_ids]  # [batch, seq_len, dim//2]
        sin = sin[position_ids]
    else:
        cos = cos[:seq_len]  # [seq_len, dim//2]
        sin = sin[:seq_len]

    # Expand for broadcasting
    while cos.ndim < x.ndim:
        cos = cos[None, ...]
        sin = sin[None, ...]

    # Split into pairs for rotation
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]

    # Apply rotation
    rotated = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    return rotated


class RotaryEmbedding(hk.Module):
    """
    Rotary Position Embedding module.

    Better than learned positional embeddings for:
    - Long sequence extrapolation
    - Relative position modeling
    - Translation equivariance
    """

    def __init__(
        self,
        dim: int,
        max_seq_length: int = 2048,
        theta: float = 10000.0,
        scaling_factor: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta
        self.scaling_factor = scaling_factor

        # Precompute frequencies
        self.cos_cached, self.sin_cached = precompute_rope_frequencies(
            dim, max_seq_length, theta, scaling_factor
        )

    def __call__(
        self, q: jnp.ndarray, k: jnp.ndarray, position_ids: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply RoPE to queries and keys."""
        q_rotated = apply_rope(q, self.cos_cached, self.sin_cached, position_ids)
        k_rotated = apply_rope(k, self.cos_cached, self.sin_cached, position_ids)
        return q_rotated, k_rotated


# =============================================================================
# Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)
# =============================================================================


class GroupedQueryAttention(hk.Module):
    """
    Grouped-Query Attention (GQA) for efficient inference.

    Instead of separate KV heads per query head, groups of query heads
    share the same KV heads, reducing KV cache size by num_heads/num_kv_heads.

    Special cases:
    - num_kv_heads = num_heads: Standard MHA
    - num_kv_heads = 1: Multi-Query Attention (MQA)
    - 1 < num_kv_heads < num_heads: GQA

    Benefits:
    - 2-4x smaller KV cache
    - Faster inference with similar quality
    - Better tensor parallelism efficiency
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float = 0.0,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        max_seq_length: int = 2048,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.use_rope = use_rope

        # Validate
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_model = num_heads * head_dim

        # RoPE if enabled
        if use_rope:
            self.rope = RotaryEmbedding(head_dim, max_seq_length, rope_theta)

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        is_training: bool = True,
        position_ids: Optional[jnp.ndarray] = None,
        kv_cache: Optional["KVPrefixCache"] = None,
        layer_id: int = 0,
        prefix_id: Optional[str] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply grouped-query attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, 1, seq_len, seq_len] or [batch, seq_len]
            is_training: Whether in training mode
            position_ids: Optional position indices
            kv_cache: Optional KVPrefixCache for prefix caching
            layer_id: Layer index for cache storage/lookup
            prefix_id: Prefix identifier for cache key

        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Projections
        q_proj = hk.Linear(self.num_heads * self.head_dim, name="q_proj")
        k_proj = hk.Linear(self.num_kv_heads * self.head_dim, name="k_proj")
        v_proj = hk.Linear(self.num_kv_heads * self.head_dim, name="v_proj")
        o_proj = hk.Linear(self.d_model, name="o_proj")

        # Project to Q, K, V
        q = q_proj(x)  # [batch, seq_len, num_heads * head_dim]
        k = k_proj(x)  # [batch, seq_len, num_kv_heads * head_dim]
        v = v_proj(x)  # [batch, seq_len, num_kv_heads * head_dim]

        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        if self.use_rope:
            # Transpose for RoPE: [batch, heads, seq_len, head_dim]
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))

            # Apply RoPE per head
            q_rope = []
            for h in range(self.num_heads):
                q_h, _ = self.rope(q[:, h : h + 1], k[:, 0:1], position_ids)
                q_rope.append(q_h)
            q = jnp.concatenate(q_rope, axis=1)

            k_rope = []
            for h in range(self.num_kv_heads):
                _, k_h = self.rope(q[:, 0:1], k[:, h : h + 1], position_ids)
                k_rope.append(k_h)
            k = jnp.concatenate(k_rope, axis=1)

            v = jnp.transpose(v, (0, 2, 1, 3))
        else:
            # Transpose: [batch, heads, seq_len, head_dim]
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

        # KV Prefix Cache: look up or store cached K/V projections
        if kv_cache is not None and prefix_id is not None:
            cached_k, cached_v = kv_cache.get(prefix_id, layer_id)
            if cached_k is not None and cached_v is not None:
                # Cache hit — prepend cached K/V to current K/V
                # cached tensors are [batch, num_kv_heads, prefix_len, head_dim]
                k = jnp.concatenate([cached_k.astype(k.dtype), k], axis=2)
                v = jnp.concatenate([cached_v.astype(v.dtype), v], axis=2)
            else:
                # Cache miss — store current K/V as prefix for future reuse
                kv_cache.store(prefix_id, layer_id, k, v)

        # Repeat KV heads for grouped attention
        if self.num_queries_per_kv > 1:
            k = jnp.repeat(k, self.num_queries_per_kv, axis=1)
            v = jnp.repeat(v, self.num_queries_per_kv, axis=1)

        # Compute attention scores
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

        # Apply mask
        if mask is not None:
            if mask.ndim == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                mask = mask[:, None, :, :]
            scores = jnp.where(mask > 0, scores, -1e9)

        # Softmax
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Dropout
        if is_training and self.dropout_rate > 0:
            attention_weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, attention_weights)

        # Apply attention to values
        attended = jnp.einsum("bhij,bhjd->bhid", attention_weights, v)

        # Reshape back
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = o_proj(attended)

        return output, attention_weights


# =============================================================================
# Sliding Window Attention
# =============================================================================


class SlidingWindowAttention(hk.Module):
    """
    Sliding Window Attention for very long sequences.

    Each token only attends to a local window of neighbors,
    reducing complexity from O(n²) to O(n * window_size).

    Good for:
    - Very long documents (>8k tokens)
    - When global attention isn't necessary
    - Memory-constrained environments
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        window_size: int = 512,
        use_rope: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.use_rope = use_rope
        self.d_model = num_heads * head_dim

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, is_training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply sliding window attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask (combined with window mask)
            is_training: Whether in training mode
        """
        batch_size, seq_len, _ = x.shape

        # Projections
        q_proj = hk.Linear(self.d_model, name="q_proj")
        k_proj = hk.Linear(self.d_model, name="k_proj")
        v_proj = hk.Linear(self.d_model, name="v_proj")
        o_proj = hk.Linear(self.d_model, name="o_proj")

        q = q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: [batch, heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Create sliding window mask
        # Each position i attends to [max(0, i-window_size+1), i+1]
        positions = jnp.arange(seq_len)
        window_mask = jnp.abs(positions[:, None] - positions[None, :]) < self.window_size
        window_mask = window_mask.astype(jnp.float32)

        # Make causal (only attend to past)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        window_mask = window_mask * causal_mask

        # Expand window_mask to [batch, heads, seq, seq]
        window_mask = window_mask[None, None, :, :]
        window_mask = jnp.broadcast_to(window_mask, (batch_size, self.num_heads, seq_len, seq_len))

        # Combine with input mask if provided
        if mask is not None:
            if mask.ndim == 2:  # [batch, seq_len]
                mask_expanded = mask[:, None, None, :] * mask[:, None, :, None]
                mask_expanded = jnp.broadcast_to(
                    mask_expanded, (batch_size, self.num_heads, seq_len, seq_len)
                )
                window_mask = window_mask * mask_expanded
            elif mask.ndim == 3:  # [batch, seq, seq]
                mask_expanded = mask[:, None, :, :]
                mask_expanded = jnp.broadcast_to(
                    mask_expanded, (batch_size, self.num_heads, seq_len, seq_len)
                )
                window_mask = window_mask * mask_expanded
            elif mask.ndim == 4:  # [batch, heads, seq, seq]
                # Broadcast to match our num_heads
                if mask.shape[1] != self.num_heads:
                    mask = jnp.broadcast_to(mask, (batch_size, self.num_heads, seq_len, seq_len))
                window_mask = window_mask * mask

        # Compute attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale

        # Apply window mask
        scores = jnp.where(window_mask > 0, scores, -1e9)

        attention_weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhij,bhjd->bhid", attention_weights, v)

        # Reshape and project
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_len, self.d_model)
        output = o_proj(attended)

        return output, attention_weights


# =============================================================================
# Linear Attention (Approximate)
# =============================================================================


class LinearAttention(hk.Module):
    """
    Linear Attention with kernel approximation.

    Replaces softmax(QK^T)V with φ(Q)φ(K)^T V where φ is a feature map,
    enabling O(n) complexity instead of O(n²).

    Trade-off: Slight quality reduction for massive speed gains on long sequences.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        feature_map: Literal["elu", "relu", "softmax"] = "elu",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_map = feature_map
        self.d_model = num_heads * head_dim

    def _apply_feature_map(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply feature map for kernel approximation."""
        if self.feature_map == "elu":
            return jax.nn.elu(x) + 1
        elif self.feature_map == "relu":
            return jax.nn.relu(x)
        else:  # softmax-like
            return jax.nn.softmax(x, axis=-1)

    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """Apply linear attention."""
        batch_size, seq_len, _ = x.shape

        q_proj = hk.Linear(self.d_model, name="q_proj")
        k_proj = hk.Linear(self.d_model, name="k_proj")
        v_proj = hk.Linear(self.d_model, name="v_proj")
        o_proj = hk.Linear(self.d_model, name="o_proj")

        q = q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply feature maps
        q = self._apply_feature_map(q)
        k = self._apply_feature_map(k)

        # Linear attention: φ(Q)(φ(K)^T V) - computed right-to-left for O(n)
        # [batch, seq, heads, head_dim] x [batch, seq, heads, head_dim] -> [batch, heads, head_dim, head_dim]
        kv = jnp.einsum("bshd,bshv->bhdv", k, v)

        # [batch, seq, heads, head_dim] x [batch, heads, head_dim, head_dim] -> [batch, seq, heads, head_dim]
        output = jnp.einsum("bshd,bhdv->bshv", q, kv)

        # Normalize
        k_sum = jnp.sum(k, axis=1, keepdims=True)  # [batch, 1, heads, head_dim]
        normalizer = jnp.einsum("bshd,bthd->bsh", q, k_sum) + 1e-6
        output = output / normalizer[..., None]

        # Reshape and project
        output = output.reshape(batch_size, seq_len, self.d_model)
        output = o_proj(output)

        return output


# =============================================================================
# Advanced Self-Attention Model
# =============================================================================


class AdvancedSelfAttention(hk.Module):
    """
    Advanced Self-Attention combining multiple techniques.

    Features:
    - RoPE for position encoding (configurable)
    - GQA/MQA for efficient inference
    - Sliding window for very long sequences
    - Spiking attention integration
    - Flash attention when available

    Backward compatible with SelfAttentionModel interface.
    """

    def __init__(self, config: AttentionConfig, vocab_size: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.vocab_size = vocab_size

        # Embedding
        self.embedding = hk.Embed(vocab_size, config.d_model, name="token_embedding")

        # Choose attention variant
        if config.attention_type == "sliding":
            self.attention = SlidingWindowAttention(
                config.num_heads, self.head_dim, config.sliding_window_size
            )
        elif config.attention_type == "linear":
            self.attention = LinearAttention(config.num_heads, self.head_dim)
        else:  # standard, gqa, mqa
            self.attention = GroupedQueryAttention(
                config.num_heads,
                self.num_kv_heads,
                self.head_dim,
                config.dropout_rate,
                use_rope=(config.position_encoding == "rope"),
                rope_theta=config.rope_theta,
                max_seq_length=config.max_seq_length,
            )

        # Layer norms
        self.norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # FFN
        self.ffn = hk.Sequential(
            [
                hk.Linear(config.d_model * 4),
                jax.nn.silu,
                hk.Linear(config.d_model),
            ]
        )

        # Output projection
        self.proj = hk.Linear(vocab_size)

        # Learned position embeddings (fallback)
        if config.position_encoding == "learned":
            self.position_embedding = hk.Embed(config.max_seq_length, config.d_model)

    def __call__(
        self,
        inputs: jnp.ndarray,
        return_attention: bool = False,
        spike_threshold: float = 0.1,
        epsilon: float = 1e-8,
        is_training: bool = True,
    ) -> Any:
        """
        Forward pass with advanced attention.

        Args:
            inputs: Input token IDs [batch, seq_len]
            return_attention: Whether to return attention weights
            spike_threshold: Threshold for spiking attention
            epsilon: Small constant for numerical stability
            is_training: Whether in training mode

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            attention_weights: (optional) [batch, num_heads, seq_len, seq_len]
        """
        inputs = jnp.asarray(inputs, dtype=jnp.int32)
        batch_size, seq_len = inputs.shape

        # Embedding
        x = self.embedding(inputs) * jnp.sqrt(self.d_model)

        # Add learned position embeddings if configured
        if self.config.position_encoding == "learned":
            positions = jnp.arange(seq_len)
            x = x + self.position_embedding(positions)

        # Create attention mask
        mask = (inputs != 0).astype(jnp.float32)

        # Causal mask for autoregressive
        if self.config.causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            mask = mask[:, None, :] * causal_mask[None, :, :]

        # Pre-norm
        x_norm = self.norm1(x)

        # Attention
        if isinstance(self.attention, LinearAttention):
            attn_out = self.attention(x_norm, is_training)
            attention_weights = None
        else:
            attn_out, attention_weights = self.attention(x_norm, mask, is_training)

        # Apply spiking (if enabled)
        if self.config.enable_spiking and spike_threshold is not None:
            if spike_threshold > 0 and spike_threshold < 1:
                spiking_mask = jnp.abs(attn_out) > spike_threshold
                attn_out = jnp.where(spiking_mask, attn_out, 0.0)
                # Renormalize
                attn_out = attn_out / (jnp.sum(jnp.abs(attn_out), axis=-1, keepdims=True) + epsilon)

        # Residual connection
        x = x + attn_out

        # FFN with pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        # Output logits
        logits = self.proj(x)

        if return_attention:
            return logits, attention_weights
        return logits


# =============================================================================
# Factory Functions
# =============================================================================


def create_attention_config(
    d_model: int = 384,
    num_heads: int = 8,
    attention_type: str = "standard",
    position_encoding: str = "rope",
    num_kv_heads: Optional[int] = None,
    **kwargs,
) -> AttentionConfig:
    """
    Create attention configuration.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        attention_type: "standard", "gqa", "mqa", "linear", "sliding"
        position_encoding: "rope", "learned", "alibi", "none"
        num_kv_heads: KV heads for GQA (None = MHA, 1 = MQA)
        **kwargs: Additional config options

    Returns:
        AttentionConfig instance
    """
    # Set defaults based on attention type
    if attention_type == "mqa":
        num_kv_heads = 1
    elif attention_type == "gqa" and num_kv_heads is None:
        num_kv_heads = max(1, num_heads // 4)  # Default: 4x reduction

    return AttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        attention_type=attention_type,
        position_encoding=position_encoding,
        **kwargs,
    )


def create_advanced_attention(config: AttentionConfig, vocab_size: int) -> AdvancedSelfAttention:
    """Factory to create AdvancedSelfAttention from src.config."""
    return AdvancedSelfAttention(config, vocab_size)


# =============================================================================
# Utility Functions
# =============================================================================


def compute_attention_flops(
    batch_size: int, seq_len: int, d_model: int, num_heads: int, attention_type: str = "standard"
) -> int:
    """
    Estimate FLOPs for attention computation.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        attention_type: Type of attention

    Returns:
        Estimated FLOPs
    """
    head_dim = d_model // num_heads

    if attention_type == "linear":
        # O(n * d²) for linear attention
        return batch_size * seq_len * d_model * head_dim * 2
    elif attention_type == "sliding":
        # O(n * window * d)
        window = 512  # Default
        return batch_size * seq_len * window * head_dim * 2
    else:
        # O(n² * d) for standard/GQA/MQA
        return batch_size * seq_len * seq_len * head_dim * 2


def estimate_kv_cache_size(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_layers: int,
    num_kv_heads: int,
    num_heads: int = 8,
    dtype_bytes: int = 2,  # bfloat16
) -> Dict[str, float]:
    """
    Estimate KV cache memory for inference.

    Args:
        batch_size: Batch size
        seq_len: Maximum sequence length
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads (for GQA)
        num_heads: Number of query heads (for computing head_dim)
        dtype_bytes: Bytes per element (2 for bf16, 4 for fp32)

    Returns:
        Dictionary with memory estimates in GB
    """
    # Head dim is based on query heads, not KV heads
    head_dim = d_model // num_heads

    # KV cache: 2 (K and V) * batch * layers * seq_len * num_kv_heads * head_dim
    cache_elements = 2 * batch_size * num_layers * seq_len * num_kv_heads * head_dim
    cache_bytes = cache_elements * dtype_bytes
    cache_gb = cache_bytes / (1024**3)

    return {
        "kv_cache_gb": cache_gb,
        "per_token_mb": (cache_bytes / seq_len) / (1024**2),
        "elements": cache_elements,
    }


# =============================================================================
# KV Prefix Cache
# =============================================================================


class KVPrefixCache:
    """
    KV Prefix Cache for efficient inference with shared prompt prefixes.

    Caches the Key and Value projections for common prompt prefixes
    (system prompts, few-shot examples, etc.) so they don't need to
    be recomputed for every new request.

    Key design choices:
    - Pure JAX arrays (no mutable state) for compatibility with jax.jit
    - Per-layer caching: each transformer layer has its own KV cache entry
    - Supports LRU, FIFO, and LFU eviction policies
    - Dtype-aware: stores in bfloat16 by default to save memory

    Usage:
        cache = KVPrefixCache(num_layers=12, max_prefix_len=256, ...)

        # During first forward pass with a new prefix:
        cache.store(prefix_id="system_v1", layer=0, keys=k, values=v)

        # During subsequent passes with same prefix:
        cached_k, cached_v = cache.get(prefix_id="system_v1", layer=0)
        if cached_k is not None:
            # Concatenate cached prefix KV with new token KV
            full_k = jnp.concatenate([cached_k, new_k], axis=1)
            full_v = jnp.concatenate([cached_v, new_v], axis=1)

    Args:
        num_layers: Number of transformer layers to cache
        max_prefix_len: Maximum prefix sequence length
        num_kv_heads: Number of KV attention heads
        head_dim: Dimension per attention head
        max_entries: Maximum number of cached prefixes
        eviction_policy: Cache eviction strategy ("lru", "fifo", "lfu")
        dtype: Storage dtype for cached tensors
    """

    def __init__(
        self,
        num_layers: int,
        max_prefix_len: int,
        num_kv_heads: int,
        head_dim: int,
        max_entries: int = 32,
        eviction_policy: str = "lru",
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.num_layers = num_layers
        self.max_prefix_len = max_prefix_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy
        self.dtype = dtype

        # Cache storage: prefix_id -> {layer_idx: (keys, values, metadata)}
        self._cache: Dict[str, Dict[int, Tuple[jnp.ndarray, jnp.ndarray]]] = {}
        # Access tracking for eviction
        self._access_count: Dict[str, int] = {}
        self._access_order: list = []  # Most recent at end
        self._insertion_order: list = []  # Oldest at front

    @property
    def size(self) -> int:
        """Number of cached prefix entries."""
        return len(self._cache)

    @property
    def is_full(self) -> bool:
        """Whether cache is at capacity."""
        return len(self._cache) >= self.max_entries

    def _evict(self) -> None:
        """Evict one entry based on eviction policy."""
        if not self._cache:
            return

        if self.eviction_policy == "lru":
            # Remove least recently used
            victim = self._access_order[0]
            self._access_order.remove(victim)
        elif self.eviction_policy == "fifo":
            # Remove oldest inserted
            victim = self._insertion_order[0]
            self._insertion_order.remove(victim)
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            victim = min(self._access_count.keys(), key=lambda k: self._access_count[k])
        else:
            # Default to LRU
            victim = self._access_order[0]
            self._access_order.remove(victim)

        del self._cache[victim]
        self._access_count.pop(victim, None)
        if victim in self._insertion_order:
            self._insertion_order.remove(victim)
        if victim in self._access_order:
            self._access_order.remove(victim)

    def _touch(self, prefix_id: str) -> None:
        """Update access tracking for a cache entry."""
        self._access_count[prefix_id] = self._access_count.get(prefix_id, 0) + 1
        if prefix_id in self._access_order:
            self._access_order.remove(prefix_id)
        self._access_order.append(prefix_id)

    def store(
        self,
        prefix_id: str,
        layer: int,
        keys: jnp.ndarray,
        values: jnp.ndarray,
    ) -> None:
        """
        Store KV projections for a prefix at a specific layer.

        Args:
            prefix_id: Unique identifier for the prefix
            layer: Transformer layer index
            keys: Key projections [batch, seq_len, num_kv_heads, head_dim]
                  or [batch, num_kv_heads, seq_len, head_dim]
            values: Value projections (same shape as keys)
        """
        # Truncate to max prefix length
        seq_axis = 2 if keys.ndim == 4 and keys.shape[1] == self.num_kv_heads else 1
        if keys.shape[seq_axis] > self.max_prefix_len:
            if seq_axis == 1:
                keys = keys[:, : self.max_prefix_len]
                values = values[:, : self.max_prefix_len]
            else:
                keys = keys[:, :, : self.max_prefix_len]
                values = values[:, :, : self.max_prefix_len]

        # Convert to storage dtype
        keys = keys.astype(self.dtype)
        values = values.astype(self.dtype)

        # Evict if at capacity and this is a new entry
        if prefix_id not in self._cache and self.is_full:
            self._evict()

        # Store
        if prefix_id not in self._cache:
            self._cache[prefix_id] = {}
            self._insertion_order.append(prefix_id)

        self._cache[prefix_id][layer] = (keys, values)
        self._touch(prefix_id)

    def get(
        self,
        prefix_id: str,
        layer: int,
    ) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """
        Retrieve cached KV projections for a prefix at a specific layer.

        Args:
            prefix_id: Unique identifier for the prefix
            layer: Transformer layer index

        Returns:
            (keys, values) if cached, (None, None) otherwise
        """
        if prefix_id not in self._cache:
            return None, None

        if layer not in self._cache[prefix_id]:
            return None, None

        self._touch(prefix_id)
        keys, values = self._cache[prefix_id][layer]
        return keys, values

    def has(self, prefix_id: str, layer: Optional[int] = None) -> bool:
        """
        Check if a prefix is cached.

        Args:
            prefix_id: Unique identifier for the prefix
            layer: If specified, check for a specific layer

        Returns:
            True if the prefix (and optionally layer) is cached
        """
        if prefix_id not in self._cache:
            return False
        if layer is not None:
            return layer in self._cache[prefix_id]
        return True

    def invalidate(self, prefix_id: str) -> bool:
        """
        Remove a specific prefix from the cache.

        Args:
            prefix_id: Unique identifier for the prefix to remove

        Returns:
            True if the entry was found and removed
        """
        if prefix_id not in self._cache:
            return False

        del self._cache[prefix_id]
        self._access_count.pop(prefix_id, None)
        if prefix_id in self._access_order:
            self._access_order.remove(prefix_id)
        if prefix_id in self._insertion_order:
            self._insertion_order.remove(prefix_id)
        return True

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_count.clear()
        self._access_order.clear()
        self._insertion_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats including size, hit rates, memory usage
        """
        total_elements = 0
        for prefix_id, layers in self._cache.items():
            for layer_idx, (k, v) in layers.items():
                total_elements += k.size + v.size

        bytes_per_element = 2 if self.dtype == jnp.bfloat16 else 4
        memory_bytes = total_elements * bytes_per_element

        return {
            "num_entries": len(self._cache),
            "max_entries": self.max_entries,
            "utilization": len(self._cache) / self.max_entries if self.max_entries > 0 else 0,
            "total_elements": total_elements,
            "memory_mb": memory_bytes / (1024**2),
            "eviction_policy": self.eviction_policy,
            "access_counts": dict(self._access_count),
        }
