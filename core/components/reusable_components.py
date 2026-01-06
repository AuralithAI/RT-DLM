"""
Reusable Components for RT-DLM AGI System

This module provides shared, reusable utilities that can be referenced
across multiple parts of the codebase to reduce code duplication and
enhance maintainability.

Key Components:
- ReusableAttention: Unified attention wrapper with spiking/pruning support
- SpikingMechanism: Modular spiking attention implementation
- PruningManager: Centralized pruning utilities for attention heads and FFN neurons
- AttentionConfig: Configuration dataclass for attention parameters
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Callable, List
import haiku as hk
import jax
import jax.numpy as jnp


@dataclass
class AttentionConfig:
    """Configuration for reusable attention components."""
    d_model: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1
    spike_threshold: float = 0.1
    epsilon: float = 1e-8
    head_pruning_threshold: float = 0.01
    ffn_pruning_threshold: float = 0.01
    enable_spiking: bool = True
    enable_pruning: bool = True
    enable_usage_tracking: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'spike_threshold': self.spike_threshold,
            'epsilon': self.epsilon,
            'head_pruning_threshold': self.head_pruning_threshold,
            'ffn_pruning_threshold': self.ffn_pruning_threshold,
            'enable_spiking': self.enable_spiking,
            'enable_pruning': self.enable_pruning,
            'enable_usage_tracking': self.enable_usage_tracking,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AttentionConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class SpikingMechanism:
    """
    Modular spiking attention implementation.
    
    Applies neuromorphic-inspired thresholding to attention scores,
    activating only important tokens above the spike threshold.
    """
    
    def __init__(self, spike_threshold: float = 0.1, epsilon: float = 1e-8):
        """
        Initialize spiking mechanism.
        
        Args:
            spike_threshold: Threshold for attention score activation (0-1)
            epsilon: Small constant for numerical stability
        """
        self.spike_threshold = spike_threshold
        self.epsilon = epsilon
    
    def apply(self, scores: jnp.ndarray) -> jnp.ndarray:
        """
        Apply spiking to attention scores.
        
        Args:
            scores: Attention scores tensor
            
        Returns:
            Spiked and renormalized attention scores
        """
        if not self._validate_threshold():
            return scores
        
        spiking_mask = scores > self.spike_threshold
        spiked_scores = jnp.where(spiking_mask, scores, 0.0)
        normalized = spiked_scores / (jnp.sum(spiked_scores, axis=-1, keepdims=True) + self.epsilon)
        return normalized
    
    def _validate_threshold(self) -> bool:
        """Validate spike threshold is in valid range."""
        return (self.spike_threshold is not None and 
                self.epsilon is not None and 
                0 <= self.spike_threshold <= 1)
    
    def update_threshold(self, new_threshold: float) -> None:
        """Update spike threshold dynamically."""
        if 0 <= new_threshold <= 1:
            self.spike_threshold = new_threshold
    
    def get_sparsity(self, scores: jnp.ndarray) -> float:
        """Calculate sparsity ratio after spiking."""
        spiking_mask = scores > self.spike_threshold
        total_elements = scores.size
        active_elements = jnp.sum(spiking_mask)
        return float(1.0 - (active_elements / total_elements))


class PruningManager:
    """
    Centralized pruning utilities for attention heads and FFN neurons.
    
    Tracks usage statistics and provides methods to prune underutilized
    components while maintaining model functionality.
    """
    
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        head_threshold: float = 0.01,
        ffn_threshold: float = 0.01
    ):
        """
        Initialize pruning manager.
        
        Args:
            num_heads: Number of attention heads
            d_model: Model dimension
            head_threshold: Minimum usage for head retention
            ffn_threshold: Minimum usage for FFN neuron retention
        """
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_threshold = head_threshold
        self.ffn_threshold = ffn_threshold
    
    def compute_head_usage(self, attn_weights: jnp.ndarray) -> jnp.ndarray:
        """
        Compute attention head usage from attention weights.
        
        Args:
            attn_weights: Attention weights tensor
            
        Returns:
            Per-head usage statistics
        """
        if attn_weights.ndim == 3:
            # Shape: [batch, seq, d_model]
            head_dim = self.d_model // self.num_heads
            head_usage = jnp.zeros((self.num_heads,))
            for head in range(self.num_heads):
                head_start = head * head_dim
                head_end = head_start + head_dim
                head_weights = attn_weights[:, :, head_start:head_end]
                head_usage = head_usage.at[head].set(jnp.mean(jnp.abs(head_weights)))
            return head_usage
        elif attn_weights.ndim == 4:
            # Shape: [batch, heads, seq, seq]
            return jnp.mean(jnp.abs(attn_weights), axis=(0, 2, 3))
        else:
            raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}")
    
    def compute_ffn_usage(self, ffn_out: jnp.ndarray) -> jnp.ndarray:
        """
        Compute FFN neuron usage from FFN output.
        
        Args:
            ffn_out: FFN output tensor
            
        Returns:
            Per-neuron usage statistics
        """
        return jnp.mean(jnp.abs(ffn_out), axis=(0, 1))
    
    def get_pruning_mask(
        self,
        head_usage: jnp.ndarray,
        ffn_usage: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get boolean masks for components to keep.
        
        Args:
            head_usage: Per-head usage statistics
            ffn_usage: Per-neuron usage statistics
            
        Returns:
            Tuple of (active_heads_mask, active_ffn_mask)
        """
        active_heads = head_usage > self.head_threshold
        active_ffn = ffn_usage > self.ffn_threshold
        
        # Ensure at least one component remains
        if jnp.sum(active_heads) < 1:
            # Keep the most used head
            max_head = jnp.argmax(head_usage)
            active_heads = active_heads.at[max_head].set(True)
        
        if jnp.sum(active_ffn) < 1:
            # Keep the most used neuron
            max_neuron = jnp.argmax(ffn_usage)
            active_ffn = active_ffn.at[max_neuron].set(True)
        
        return active_heads, active_ffn
    
    def compute_compression_ratio(
        self,
        active_heads: jnp.ndarray,
        active_ffn: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Calculate compression achieved by pruning.
        
        Args:
            active_heads: Boolean mask for active heads
            active_ffn: Boolean mask for active FFN neurons
            
        Returns:
            Dictionary with compression statistics
        """
        head_compression = 1.0 - (jnp.sum(active_heads) / self.num_heads)
        ffn_compression = 1.0 - (jnp.sum(active_ffn) / self.d_model)
        return {
            'head_compression': float(head_compression),
            'ffn_compression': float(ffn_compression),
            'overall_compression': float((head_compression + ffn_compression) / 2),
            'active_heads': int(jnp.sum(active_heads)),
            'active_neurons': int(jnp.sum(active_ffn)),
        }


class ReusableAttention(hk.Module):
    """
    Unified attention wrapper with spiking and pruning support.
    
    This class provides a reusable attention mechanism that can be
    shared across multiple components (TMS, self-attention, transformers)
    while maintaining consistent spiking/pruning behavior.
    
    Features:
    - Multi-head attention with configurable heads and dimensions
    - Spiking attention for neuromorphic-style sparse activations
    - Usage tracking for dynamic pruning
    - Configurable via AttentionConfig dataclass
    
    Example:
        config = AttentionConfig(d_model=256, num_heads=8)
        attention = ReusableAttention(config, name="shared_attention")
        output = attention(inputs, spike_threshold=0.1, epsilon=1e-8)
    """
    
    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        name: Optional[str] = None
    ):
        """
        Initialize reusable attention.
        
        Args:
            config: AttentionConfig instance (preferred)
            d_model: Model dimension (alternative to config)
            num_heads: Number of attention heads (alternative to config)
            name: Module name for Haiku
        """
        super().__init__(name=name)
        
        # Use config or individual params
        if config is not None:
            self.d_model = config.d_model
            self.num_heads = config.num_heads
            self.default_spike_threshold = config.spike_threshold
            self.default_epsilon = config.epsilon
            self.enable_spiking = config.enable_spiking
            self.enable_usage_tracking = config.enable_usage_tracking
        else:
            self.d_model = d_model or 256
            self.num_heads = num_heads or 8
            self.default_spike_threshold = 0.1
            self.default_epsilon = 1e-8
            self.enable_spiking = True
            self.enable_usage_tracking = True
        
        # Core attention mechanism
        self.attention = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.d_model // self.num_heads,
            model_size=self.d_model,
            w_init=hk.initializers.VarianceScaling(1.0),
        )
        
        # Layer normalization
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
        # Spiking mechanism
        self.spiking = SpikingMechanism(
            self.default_spike_threshold,
            self.default_epsilon
        )
        
        # Usage tracking state
        if self.enable_usage_tracking:
            self.head_usage = hk.get_state(
                "head_usage",
                [self.num_heads],
                dtype=jnp.float32,
                init=jnp.zeros
            )
    
    def apply_attention(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply raw multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output
        """
        return self.attention(query=query, key=key, value=value, mask=mask)
    
    def apply_reusable_attention(
        self,
        inputs: jnp.ndarray,
        spike_threshold: Optional[float] = None,
        epsilon: Optional[float] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Convenience method for self-attention with spiking.
        
        Args:
            inputs: Input tensor [batch, seq, d_model]
            spike_threshold: Override default spike threshold
            epsilon: Override default epsilon
            
        Returns:
            Attention output with spiking applied (output, None)
        """
        return self.__call__(
            inputs,
            spike_threshold=spike_threshold,
            epsilon=epsilon,
            return_attention=False
        )
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        return_attention: bool = False,
        spike_threshold: Optional[float] = None,
        epsilon: Optional[float] = None,
        mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Forward pass with optional spiking attention.
        
        Args:
            inputs: Input tensor [batch, seq, d_model]
            return_attention: Whether to return attention weights
            spike_threshold: Threshold for spiking (uses default if None)
            epsilon: Numerical stability constant
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights or None)
        """
        # Use defaults if not specified
        threshold = spike_threshold if spike_threshold is not None else self.default_spike_threshold
        eps = epsilon if epsilon is not None else self.default_epsilon
        
        # Update spiking mechanism if needed
        if threshold != self.spiking.spike_threshold:
            self.spiking.update_threshold(threshold)
        if eps != self.spiking.epsilon:
            self.spiking.epsilon = eps
        
        # Normalize input
        x = self.norm(inputs)
        
        # Apply multi-head attention
        attn_out = self.attention(query=x, key=x, value=x, mask=mask)
        
        # Apply spiking if enabled
        if self.enable_spiking:
            attn_out = self.spiking.apply(attn_out)
        
        # Residual connection
        output = inputs + attn_out
        
        if return_attention:
            # Compute attention weights for tracking
            attention_weights = jax.nn.softmax(attn_out, axis=-1)
            
            # Update usage tracking
            if self.enable_usage_tracking:
                self._update_usage(attention_weights)
            
            return output, attention_weights
        
        return output, None
    
    def _update_usage(self, attn_weights: jnp.ndarray) -> None:
        """Update head usage statistics."""
        pruning_manager = PruningManager(
            self.num_heads,
            self.d_model
        )
        head_usage_update = pruning_manager.compute_head_usage(attn_weights)
        
        current_usage = hk.get_state(
            "head_usage",
            [self.num_heads],
            dtype=jnp.float32,
            init=jnp.zeros
        )
        hk.set_state("head_usage", current_usage + head_usage_update)
    
    def get_head_usage(self) -> jnp.ndarray:
        """Get current head usage statistics."""
        return hk.get_state(
            "head_usage",
            [self.num_heads],
            dtype=jnp.float32,
            init=jnp.zeros
        )


class ReusableFeedForward(hk.Module):
    """
    Reusable feed-forward network with usage tracking.
    
    Provides a shared FFN implementation that can be used across
    different model components with consistent behavior.
    """
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        activation: Callable = jax.nn.silu,
        name: Optional[str] = None
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            expansion_factor: Hidden layer expansion
            activation: Activation function
            name: Module name
        """
        super().__init__(name=name)
        self.d_model = d_model
        self.hidden_dim = d_model * expansion_factor
        self.activation = activation
        
        self.ffn = hk.Sequential([
            hk.Linear(self.hidden_dim, w_init=hk.initializers.VarianceScaling(1.0)),
            activation,
            hk.Linear(d_model, w_init=hk.initializers.VarianceScaling(1.0)),
        ])
        
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
        # Usage tracking
        self.ffn_usage = hk.get_state(
            "ffn_usage",
            [d_model],
            dtype=jnp.float32,
            init=jnp.zeros
        )
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        track_usage: bool = True
    ) -> jnp.ndarray:
        """
        Forward pass through FFN.
        
        Args:
            inputs: Input tensor
            track_usage: Whether to track neuron usage
            
        Returns:
            FFN output with residual connection
        """
        x = self.norm(inputs)
        ffn_out = self.ffn(x)
        
        if track_usage:
            self._update_usage(ffn_out)
        
        return inputs + ffn_out
    
    def _update_usage(self, ffn_out: jnp.ndarray) -> None:
        """Update FFN usage statistics."""
        usage_update = jnp.mean(jnp.abs(ffn_out), axis=(0, 1))
        current_usage = hk.get_state(
            "ffn_usage",
            [self.d_model],
            dtype=jnp.float32,
            init=jnp.zeros
        )
        hk.set_state("ffn_usage", current_usage + usage_update)


class ReusableTransformerBlock(hk.Module):
    """
    Complete transformer block combining attention and FFN.
    
    Provides a full transformer layer that can be stacked
    for building deeper architectures with consistent behavior.
    """
    
    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        name: Optional[str] = None
    ):
        """
        Initialize transformer block.
        
        Args:
            config: AttentionConfig instance
            d_model: Model dimension
            num_heads: Number of attention heads
            ffn_expansion: FFN expansion factor
            name: Module name
        """
        super().__init__(name=name)
        
        actual_d_model = config.d_model if config else d_model
        actual_num_heads = config.num_heads if config else num_heads
        
        self.attention = ReusableAttention(
            config=config,
            d_model=actual_d_model,
            num_heads=actual_num_heads,
            name=f"{name}_attention" if name else "attention"
        )
        
        self.ffn = ReusableFeedForward(
            d_model=actual_d_model,
            expansion_factor=ffn_expansion,
            name=f"{name}_ffn" if name else "ffn"
        )
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        spike_threshold: Optional[float] = None,
        epsilon: Optional[float] = None,
        return_attention: bool = False
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Forward pass through transformer block.
        
        Args:
            inputs: Input tensor
            spike_threshold: Spiking threshold
            epsilon: Numerical stability
            return_attention: Return attention weights
            
        Returns:
            Output tensor, optionally with attention weights
        """
        if return_attention:
            x, attn_weights = self.attention(
                inputs,
                spike_threshold=spike_threshold,
                epsilon=epsilon,
                return_attention=True
            )
        else:
            x = self.attention(
                inputs,
                spike_threshold=spike_threshold,
                epsilon=epsilon,
                return_attention=False
            )
            attn_weights = None
        
        output = self.ffn(x)
        
        if return_attention:
            return output, attn_weights
        return output


# Factory functions for easy instantiation
def create_attention(
    d_model: int = 256,
    num_heads: int = 8,
    spike_threshold: float = 0.1,
    name: Optional[str] = None
) -> ReusableAttention:
    """
    Factory function to create ReusableAttention instance.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        spike_threshold: Default spike threshold
        name: Module name
        
    Returns:
        Configured ReusableAttention instance
    """
    config = AttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        spike_threshold=spike_threshold
    )
    return ReusableAttention(config=config, name=name)


def create_transformer_block(
    d_model: int = 256,
    num_heads: int = 8,
    ffn_expansion: int = 4,
    spike_threshold: float = 0.1,
    name: Optional[str] = None
) -> ReusableTransformerBlock:
    """
    Factory function to create ReusableTransformerBlock instance.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        ffn_expansion: FFN expansion factor
        spike_threshold: Default spike threshold
        name: Module name
        
    Returns:
        Configured ReusableTransformerBlock instance
    """
    config = AttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        spike_threshold=spike_threshold
    )
    return ReusableTransformerBlock(
        config=config,
        ffn_expansion=ffn_expansion,
        name=name
    )


# Utility functions
def apply_shared_spiking(
    scores: jnp.ndarray,
    spike_threshold: float = 0.1,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Standalone spiking function for use anywhere in codebase.
    
    Args:
        scores: Attention scores
        spike_threshold: Activation threshold
        epsilon: Numerical stability
        
    Returns:
        Spiked attention scores
    """
    mechanism = SpikingMechanism(spike_threshold, epsilon)
    return mechanism.apply(scores)


def compute_attention_sparsity(
    scores: jnp.ndarray,
    spike_threshold: float = 0.1
) -> float:
    """
    Compute sparsity ratio for attention scores.
    
    Args:
        scores: Attention scores
        spike_threshold: Threshold for sparsity calculation
        
    Returns:
        Sparsity ratio (0-1)
    """
    mechanism = SpikingMechanism(spike_threshold)
    return mechanism.get_sparsity(scores)


__all__ = [
    # Configuration
    'AttentionConfig',
    # Core modules
    'ReusableAttention',
    'ReusableFeedForward',
    'ReusableTransformerBlock',
    # Mechanisms
    'SpikingMechanism',
    'PruningManager',
    # Factory functions
    'create_attention',
    'create_transformer_block',
    # Utility functions
    'apply_shared_spiking',
    'compute_attention_sparsity',
]
