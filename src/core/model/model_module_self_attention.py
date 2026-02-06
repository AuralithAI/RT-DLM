"""
Unified Self-Attention Module for RT-DLM AGI

This module provides the primary attention mechanism used throughout RT-DLM.
It consolidates all attention variants (standard MHA, GQA, RoPE, sliding window, etc.)
into a single configurable class.

Features:
- RoPE (Rotary Position Embedding): Better long-context handling
- GQA (Grouped-Query Attention): 2-4x KV cache reduction
- MQA (Multi-Query Attention): Maximum efficiency with shared KV
- Sliding Window: O(n) complexity for very long sequences
- Linear Attention: Approximate attention with O(n) complexity
- Spiking Attention: Sparse activation for efficiency
- Pruning: Dynamic head/FFN pruning based on usage

All advanced attention features are imported from advanced_attention.py
and exposed through a unified interface for use in TMSModel and other components.
"""

import haiku as hk
import jax
import os
import sys
import jax.numpy as jnp
from typing import Optional, Tuple, Any, Literal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import shared spiking/pruning utilities from reusable components
from src.core.components.reusable_components import SpikingMechanism, PruningManager

# Import advanced attention components
from src.core.model.advanced_attention import (
    AttentionConfig,
    RotaryEmbedding,
    GroupedQueryAttention,
    SlidingWindowAttention,
    LinearAttention,
)


class SelfAttentionModel(hk.Module):
    """
    Unified Self-Attention Model with advanced features.
    
    This is the primary attention module used throughout RT-DLM AGI.
    It supports multiple attention variants configured via parameters:
    
    Attention Types:
    - "standard": Traditional Multi-Head Attention
    - "gqa": Grouped-Query Attention (fewer KV heads)
    - "mqa": Multi-Query Attention (single KV head)
    - "sliding": Sliding Window Attention for long sequences
    - "linear": Linear Attention with O(n) complexity
    
    Position Encoding:
    - "rope": Rotary Position Embedding (recommended)
    - "learned": Traditional learned positional embeddings
    - "none": No positional encoding
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        vocab_size: Vocabulary size for embedding
        max_seq_length: Maximum sequence length
        attention_type: Type of attention mechanism
        num_kv_heads: Number of KV heads for GQA (None = MHA)
        position_encoding: Type of position encoding
        use_spiking: Enable spiking attention
        sliding_window_size: Window size for sliding attention
        name: Module name
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        vocab_size: int, 
        max_seq_length: int,
        attention_type: Literal["standard", "gqa", "mqa", "sliding", "linear"] = "standard",
        num_kv_heads: Optional[int] = None,
        position_encoding: Literal["rope", "learned", "none"] = "rope",
        use_spiking: bool = True,
        sliding_window_size: int = 512,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.attention_type = attention_type
        self.position_encoding = position_encoding
        self.use_spiking = use_spiking
        self.dropout_rate = dropout_rate
        
        # Determine num_kv_heads based on attention type
        if attention_type == "mqa":
            self.num_kv_heads = 1
        elif attention_type == "gqa":
            self.num_kv_heads = num_kv_heads or max(1, num_heads // 4)
        else:
            self.num_kv_heads = num_heads
        
        # Embedding layer
        self.embedding = hk.Embed(vocab_size=vocab_size, embed_dim=d_model, name="token_embedding")
        
        # Position encoding
        if position_encoding == "learned":
            self.position_embedding = hk.Embed(max_seq_length, d_model, name="position_embedding")
        elif position_encoding == "rope":
            self.rope = RotaryEmbedding(self.head_dim, max_seq_length)
        
        # Choose attention implementation based on type
        if attention_type in ["standard", "gqa", "mqa"]:
            self.attention = GroupedQueryAttention(
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dropout_rate=dropout_rate,
                use_rope=(position_encoding == "rope"),
                max_seq_length=max_seq_length
            )
        elif attention_type == "sliding":
            self.attention = SlidingWindowAttention(
                num_heads=num_heads,
                head_dim=self.head_dim,
                window_size=sliding_window_size,
                use_rope=(position_encoding == "rope")
            )
        elif attention_type == "linear":
            self.attention = LinearAttention(
                num_heads=num_heads,
                head_dim=self.head_dim
            )
        
        # Layer norms (pre-norm architecture)
        self.norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
        # FFN with SiLU activation (like LLaMA/Mistral)
        self.ffn = hk.Sequential([
            hk.Linear(d_model * 4, w_init=hk.initializers.VarianceScaling(1.0)),
            jax.nn.silu,
            hk.Linear(d_model, w_init=hk.initializers.VarianceScaling(1.0)),
        ])
        
        # Output projection
        self.proj = hk.Linear(vocab_size)
        
        # Usage tracking state for pruning
        self.head_usage = hk.get_state("head_usage", [num_heads], dtype=jnp.float32, init=jnp.zeros)
        self.ffn_usage = hk.get_state("ffn_usage", [d_model], dtype=jnp.float32, init=jnp.zeros)
        
        # Shared spiking mechanism from reusable_components
        self._spiking = SpikingMechanism(spike_threshold=0.1, epsilon=1e-8)
        
        # Shared pruning manager for usage tracking
        self._pruning_manager = PruningManager(
            num_heads=num_heads,
            d_model=d_model,
            head_threshold=0.01,
            ffn_threshold=0.01
        )

    def apply_spiking_attention(self, scores: jnp.ndarray, spike_threshold: float, epsilon: float) -> jnp.ndarray:
        """
        Apply Spiking Attention by thresholding attention scores.
        
        Delegates to shared SpikingMechanism from src.core.components.reusable_components.
        
        Args:
            scores: Attention output to apply spiking to
            spike_threshold: Threshold for spiking (0-1)
            epsilon: Small constant for numerical stability
            
        Returns:
            Spiked attention output
        """
        if not self.use_spiking:
            return scores
            
        if spike_threshold is None or epsilon is None or not 0 <= spike_threshold <= 1:
            return scores
        
        # Update shared mechanism with current parameters
        self._spiking.spike_threshold = spike_threshold
        self._spiking.epsilon = epsilon
        
        return self._spiking.apply(scores)

    def update_usage(self, attn_weights: jnp.ndarray, ffn_out: jnp.ndarray) -> None:
        """
        Update usage statistics for attention heads and FFN neurons.
        
        Delegates usage computation to shared PruningManager.
        
        Args:
            attn_weights: Attention weights [batch, heads, seq, seq]
            ffn_out: FFN output [batch, seq, d_model]
        """
        # Use shared PruningManager for head usage computation
        head_usage_update = self._pruning_manager.compute_head_usage(attn_weights)
        
        # Use shared PruningManager for FFN usage computation
        ffn_usage_update = self._pruning_manager.compute_ffn_usage(ffn_out)
        
        # Accumulate to Haiku state
        current_head_usage = hk.get_state("head_usage", [self.num_heads], dtype=jnp.float32, init=jnp.zeros)
        current_ffn_usage = hk.get_state("ffn_usage", [self.d_model], dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("head_usage", current_head_usage + head_usage_update)
        hk.set_state("ffn_usage", current_ffn_usage + ffn_usage_update)

    def prune_heads_and_ffn(self, head_threshold: float = 0.01, ffn_threshold: float = 0.01) -> "SelfAttentionModel":
        """
        Prune attention heads and FFN neurons with usage below thresholds.
        
        Returns a new model with pruned components.
        
        Args:
            head_threshold: Minimum usage to keep a head
            ffn_threshold: Minimum usage to keep a FFN neuron
            
        Returns:
            New SelfAttentionModel with pruned components
        """
        # Get usage statistics from Haiku state
        usage_heads = hk.get_state("head_usage", [self.num_heads], dtype=jnp.float32, init=jnp.zeros)
        usage_ffn = hk.get_state("ffn_usage", [self.d_model], dtype=jnp.float32, init=jnp.zeros)
        
        # Update pruning manager thresholds and get masks
        self._pruning_manager.head_threshold = head_threshold
        self._pruning_manager.ffn_threshold = ffn_threshold
        active_heads, active_ffn = self._pruning_manager.get_pruning_mask(usage_heads, usage_ffn)

        new_num_heads = int(jnp.sum(active_heads))
        new_d_model = int(jnp.sum(active_ffn))
        
        if new_num_heads == self.num_heads and new_d_model == self.d_model:
            return self

        # Create pruned model with same attention type
        new_model = SelfAttentionModel(
            d_model=new_d_model,
            num_heads=new_num_heads,
            vocab_size=self.vocab_size,
            max_seq_length=self.max_seq_length,
            attention_type=self.attention_type,
            position_encoding=self.position_encoding,
            use_spiking=self.use_spiking,
            name=self.name
        )

        return new_model
    
    def get_attention_info(self) -> dict:
        """
        Get information about the attention configuration.
        
        Returns:
            Dictionary with attention configuration details
        """
        return {
            "attention_type": self.attention_type,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "position_encoding": self.position_encoding,
            "use_spiking": self.use_spiking,
            "kv_cache_reduction": self.num_heads / self.num_kv_heads if self.num_kv_heads else 1.0,
        }

    def __call__(
        self, 
        inputs: jnp.ndarray, 
        return_attention: bool = False, 
        spike_threshold: float = 0.1, 
        epsilon: float = 1e-8,
        is_training: bool = True,
        return_hidden_states: bool = False
    ) -> Any:
        """
        Forward pass with advanced attention and spiking.
        
        Args:
            inputs: Input token IDs [batch, seq_len]
            return_attention: Whether to return attention weights
            spike_threshold: Threshold for spiking attention
            epsilon: Small constant for numerical stability
            is_training: Whether in training mode
            return_hidden_states: If True, return hidden states instead of logits
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size] (or hidden states if return_hidden_states)
            attention_weights: (optional) Attention weights
        """
        inputs = jnp.asarray(inputs, dtype=jnp.int32)
        _, seq_len = inputs.shape
        
        # Token embedding with scaling
        x = self.embedding(inputs) * jnp.sqrt(self.d_model)
        
        # Add position encoding if using learned positions
        if self.position_encoding == "learned":
            positions = jnp.arange(seq_len)
            x = x + self.position_embedding(positions)
        
        # Create attention mask
        mask = (inputs != 0).astype(jnp.float32)
        
        # Pre-norm
        x_norm = self.norm1(x)
        
        # Apply attention based on type
        if isinstance(self.attention, LinearAttention):
            attn_out = self.attention(x_norm, is_training)
            attention_weights = None
        else:
            attn_out, attention_weights = self.attention(x_norm, mask, is_training)
        
        # Apply spiking attention
        spiked_attention = self.apply_spiking_attention(attn_out, spike_threshold, epsilon)
        
        # Residual connection
        x = x + spiked_attention
        
        # FFN with pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        # Return hidden states if requested (for use in TMSModel pipeline)
        if return_hidden_states:
            if return_attention:
                return x, attention_weights
            return x
        
        # Output projection to logits
        logits = self.proj(x)

        if return_attention:
            if attention_weights is not None:
                self.update_usage(attention_weights, ffn_out)
            return logits, attention_weights
        return logits


# =============================================================================
# Factory Functions for Easy Configuration
# =============================================================================

def create_self_attention(
    d_model: int,
    num_heads: int,
    vocab_size: int,
    max_seq_length: int,
    preset: Literal["standard", "efficient", "long_context", "fast"] = "standard",
    **kwargs
) -> SelfAttentionModel:
    """
    Factory function to create SelfAttentionModel with common presets.
    
    Presets:
    - "standard": Traditional MHA with RoPE (good balance)
    - "efficient": GQA with 4x KV reduction (inference optimized)
    - "long_context": Sliding window attention (for >8k tokens)
    - "fast": Linear attention (fastest, approximate)
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        preset: Configuration preset
        **kwargs: Additional arguments passed to SelfAttentionModel
        
    Returns:
        Configured SelfAttentionModel
    """
    preset_configs = {
        "standard": {
            "attention_type": "standard",
            "position_encoding": "rope",
        },
        "efficient": {
            "attention_type": "gqa",
            "num_kv_heads": max(1, num_heads // 4),
            "position_encoding": "rope",
        },
        "long_context": {
            "attention_type": "sliding",
            "sliding_window_size": 512,
            "position_encoding": "rope",
        },
        "fast": {
            "attention_type": "linear",
            "position_encoding": "none",
        },
    }
    
    config = preset_configs.get(preset, preset_configs["standard"])
    config.update(kwargs)
    
    return SelfAttentionModel(
        d_model=d_model,
        num_heads=num_heads,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        **config
    )

