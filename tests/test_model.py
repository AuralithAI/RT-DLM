"""
Core model components for RT-DLM AGI

This module provides basic building blocks used by the test_example tests.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional


class SelfAttention(hk.Module):
    """Multi-head self attention module"""
    
    def __init__(self, d_model: int, num_heads: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply multi-head self attention.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = hk.Linear(3 * self.d_model, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(x.dtype)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = hk.Linear(self.d_model, name="out_proj")(attn_output)
        
        return output


class Embedding(hk.Module):
    """Token and positional embedding module"""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 512, 
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
    def __call__(self, token_ids: jnp.ndarray, seq_length: Optional[int] = None) -> jnp.ndarray:
        """Embed tokens with positional encoding.
        
        Args:
            token_ids: Integer tensor of shape [batch, seq_len]
            seq_length: Optional sequence length (if not provided, inferred from input)
            
        Returns:
            Embedded tensor of shape [batch, seq_len, d_model]
        """
        batch_size = token_ids.shape[0]
        seq_len = seq_length if seq_length is not None else token_ids.shape[1]
        
        # Token embedding
        token_embed = hk.Embed(vocab_size=self.vocab_size, embed_dim=self.d_model, 
                               name="token_embed")(token_ids)
        
        # Positional encoding (learnable)
        positions = jnp.arange(seq_len)
        pos_embed = hk.Embed(vocab_size=self.max_seq_len, embed_dim=self.d_model,
                            name="pos_embed")(positions)
        
        # Broadcast positional embedding to batch
        pos_embed = jnp.broadcast_to(pos_embed, (batch_size, seq_len, self.d_model))
        
        return token_embed + pos_embed


class TransformerBlock(hk.Module):
    """Single transformer block with self-attention and feedforward"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: Optional[int] = None,
                 dropout_rate: float = 0.1, name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or 4 * d_model
        self.dropout_rate = dropout_rate
        
    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """Apply transformer block.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            is_training: Whether in training mode (for dropout)
            
        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        # Self-attention with residual
        attn = SelfAttention(self.d_model, self.num_heads, name="self_attn")
        attn_out = attn(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln1")(x + attn_out)
        
        # Feedforward with residual
        ff = hk.Sequential([
            hk.Linear(self.d_ff),
            jax.nn.gelu,
            hk.Linear(self.d_model),
        ], name="ff")
        ff_out = ff(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln2")(x + ff_out)
        
        return x


class MixtureOfExperts(hk.Module):
    """Mixture of Experts layer with top-k routing"""
    
    def __init__(self, d_model: int, num_experts: int = 8, expert_dim: Optional[int] = None,
                 top_k: int = 2, name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_dim = expert_dim or 4 * d_model
        self.top_k = top_k
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply mixture of experts.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Router: compute expert scores
        router_logits = hk.Linear(self.num_experts, name="router")(x)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = jax.lax.top_k(router_probs, self.top_k)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (jnp.sum(top_k_probs, axis=-1, keepdims=True) + 1e-8)
        
        # Create expert outputs (simplified: use single linear layer per expert)
        expert_outputs = []
        for i in range(self.num_experts):
            expert = hk.Sequential([
                hk.Linear(self.expert_dim, name=f"expert_{i}_up"),
                jax.nn.gelu,
                hk.Linear(self.d_model, name=f"expert_{i}_down"),
            ])
            expert_outputs.append(expert(x))
        
        # Stack expert outputs: [num_experts, batch, seq, d_model]
        expert_outputs = jnp.stack(expert_outputs, axis=0)
        
        # Gather top-k expert outputs and combine
        # Simplified: weighted average of all experts based on router probs
        combined = jnp.einsum('ebsd,bse->bsd', expert_outputs, router_probs)
        
        return combined


# Alias for backward compatibility
EmbeddingLayer = Embedding