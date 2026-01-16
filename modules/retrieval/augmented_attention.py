"""
Retrieval-Augmented Attention

Cross-attention mechanism that integrates retrieved documents into the
forward pass. This makes retrieval differentiable - the model learns
to use retrieved information effectively.

Architecture:
    Query (model hidden states) attends to Keys/Values (retrieved docs)
    This is how industry models handle retrieval at inference.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import haiku as hk

logger = logging.getLogger(__name__)


class CrossAttentionRetrieval(hk.Module):
    """
    Cross-attention layer for retrieval augmentation.
    
    The model's hidden states (queries) attend to retrieved document
    embeddings (keys/values). This allows the model to:
    - Selectively focus on relevant retrieved information
    - Learn which retrieved content is useful
    - Integrate external knowledge smoothly
    
    Used by:
    - RETRO (DeepMind)
    - Atlas (Meta)
    - Retrieval-augmented transformers
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize cross-attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            use_bias: Use bias in projections
            name: Module name
        """
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
    
    def __call__(
        self,
        query: jnp.ndarray,
        retrieved_context: jnp.ndarray,
        retrieved_mask: Optional[jnp.ndarray] = None,
        is_training: bool = False,
    ) -> jnp.ndarray:
        """
        Apply cross-attention to retrieved context.
        
        Args:
            query: Model hidden states [batch, seq_len, d_model]
            retrieved_context: Retrieved doc embeddings [batch, num_retrieved, d_model]
            retrieved_mask: Mask for retrieved docs [batch, num_retrieved]
            is_training: Whether in training mode
            
        Returns:
            Attended output [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.shape
        _, num_retrieved, _ = retrieved_context.shape
        
        # Project queries from model hidden states
        q = hk.Linear(self.d_model, with_bias=self.use_bias, name="query")(query)
        
        # Project keys and values from retrieved context
        k = hk.Linear(self.d_model, with_bias=self.use_bias, name="key")(retrieved_context)
        v = hk.Linear(self.d_model, with_bias=self.use_bias, name="value")(retrieved_context)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, num_retrieved, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, num_retrieved, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale
        
        # Apply mask if provided
        if retrieved_mask is not None:
            # Expand mask for heads and query positions
            mask = retrieved_mask[:, None, None, :]  # [batch, 1, 1, num_retrieved]
            attention_scores = jnp.where(mask, attention_scores, -1e9)
        
        # Softmax
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        # Dropout
        if is_training and self.dropout_rate > 0:
            attention_weights = hk.dropout(
                hk.next_rng_key(), 
                self.dropout_rate, 
                attention_weights
            )
        
        # Apply attention to values
        attended = jnp.matmul(attention_weights, v)
        
        # Reshape back: [batch, seq, d_model]
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = hk.Linear(self.d_model, with_bias=self.use_bias, name="output")(attended)
        
        return output


class RetrievalAugmentedAttention(hk.Module):
    """
    Full retrieval-augmented attention block.
    
    Combines:
    - Standard self-attention
    - Cross-attention to retrieved documents
    - Gating mechanism to control retrieval influence
    
    This is the production pattern for retrieval augmentation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        retrieval_gate_init: float = 0.0,
        name: Optional[str] = None,
    ):
        """
        Initialize retrieval-augmented attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            retrieval_gate_init: Initial value for retrieval gate (0 = no retrieval)
            name: Module name
        """
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.retrieval_gate_init = retrieval_gate_init
    
    def __call__(
        self,
        x: jnp.ndarray,
        retrieved_context: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        retrieved_mask: Optional[jnp.ndarray] = None,
        is_training: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Forward pass with optional retrieval augmentation.
        
        Args:
            x: Input hidden states [batch, seq_len, d_model]
            retrieved_context: Retrieved doc embeddings [batch, num_retrieved, d_model]
            attention_mask: Self-attention mask
            retrieved_mask: Mask for retrieved docs
            is_training: Whether in training mode
            
        Returns:
            Tuple of (output, auxiliary_info)
        """
        aux_info = {}
        
        # Standard self-attention (using Haiku's built-in)
        self_attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.d_model // self.num_heads,
            w_init_scale=1.0,
            name="self_attention",
        )
        
        # Apply self-attention
        self_attn_output = self_attn(x, x, x)
        
        # Layer norm after self-attention
        self_attn_output = hk.LayerNorm(
            axis=-1, 
            create_scale=True, 
            create_offset=True, 
            name="self_attn_norm"
        )(self_attn_output)
        
        # Residual connection
        x = x + self_attn_output
        
        # Cross-attention to retrieved context (if provided)
        if retrieved_context is not None:
            cross_attn = CrossAttentionRetrieval(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name="cross_attention",
            )
            
            cross_attn_output = cross_attn(
                query=x,
                retrieved_context=retrieved_context,
                retrieved_mask=retrieved_mask,
                is_training=is_training,
            )
            
            # Learnable gate to control retrieval influence
            # Initialized to small value so model starts without relying on retrieval
            gate = hk.get_parameter(
                "retrieval_gate",
                shape=[1],
                init=hk.initializers.Constant(self.retrieval_gate_init),
            )
            gate = jax.nn.sigmoid(gate)
            
            # Gated residual
            x = x + gate * cross_attn_output
            
            aux_info["retrieval_gate"] = gate
            
        # Final layer norm
        x = hk.LayerNorm(
            axis=-1, 
            create_scale=True, 
            create_offset=True, 
            name="output_norm"
        )(x)
        
        return x, aux_info


class RetrievalAugmentedBlock(hk.Module):
    """
    Complete transformer block with retrieval augmentation.
    
    Includes:
    - Retrieval-augmented attention
    - Feed-forward network
    - Residual connections and layer norms
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        name: Optional[str] = None,
    ):
        """
        Initialize retrieval-augmented block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension (default: 4 * d_model)
            dropout_rate: Dropout rate
            name: Module name
        """
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * d_model
        self.dropout_rate = dropout_rate
    
    def __call__(
        self,
        x: jnp.ndarray,
        retrieved_context: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        retrieved_mask: Optional[jnp.ndarray] = None,
        is_training: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Forward pass."""
        aux_info = {}
        
        # Retrieval-augmented attention
        attn_block = RetrievalAugmentedAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name="attention",
        )
        
        attn_output, attn_aux = attn_block(
            x=x,
            retrieved_context=retrieved_context,
            attention_mask=attention_mask,
            retrieved_mask=retrieved_mask,
            is_training=is_training,
        )
        aux_info.update(attn_aux)
        
        # Residual
        x = x + attn_output
        
        # Feed-forward network
        ff_output = hk.Linear(self.ff_dim, name="ff_in")(x)
        ff_output = jax.nn.gelu(ff_output)
        ff_output = hk.Linear(self.d_model, name="ff_out")(ff_output)
        
        if is_training and self.dropout_rate > 0:
            ff_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, ff_output)
        
        # Residual and norm
        x = x + ff_output
        x = hk.LayerNorm(
            axis=-1, 
            create_scale=True, 
            create_offset=True, 
            name="ff_norm"
        )(x)
        
        return x, aux_info


def create_retrieval_augmented_forward(
    base_forward_fn,
    retrieval_config: Dict[str, Any],
):
    """
    Wrap a base forward function with retrieval augmentation.
    
    This factory function creates a new forward function that:
    1. Runs the base model
    2. Optionally retrieves relevant documents
    3. Applies cross-attention to retrieved content
    
    Args:
        base_forward_fn: Original model forward function
        retrieval_config: Configuration for retrieval
        
    Returns:
        Augmented forward function
    """
    def augmented_forward(
        inputs: jnp.ndarray,
        retriever=None,
        embedding_fn=None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Forward pass with optional retrieval augmentation.
        """
        aux_info = {}
        
        # Run base model
        base_output = base_forward_fn(inputs, **kwargs)
        
        # Handle tuple output (output, aux)
        if isinstance(base_output, tuple):
            hidden_states, base_aux = base_output
            aux_info.update(base_aux)
        else:
            hidden_states = base_output
        
        # Retrieval augmentation (if retriever provided)
        if retriever is not None and embedding_fn is not None:
            # Retrieve documents using query embeddings
            # In practice, retrieval happens outside JAX JIT
            # Here we assume retrieved_context is passed in kwargs
            retrieved_context = kwargs.get("retrieved_context")
            
            if retrieved_context is not None:
                # Apply cross-attention
                cross_attn = CrossAttentionRetrieval(
                    d_model=hidden_states.shape[-1],
                    num_heads=retrieval_config.get("num_heads", 8),
                    name="retrieval_cross_attn",
                )
                
                attended = cross_attn(
                    query=hidden_states,
                    retrieved_context=retrieved_context,
                    is_training=kwargs.get("is_training", False),
                )
                
                # Combine with gating
                gate = retrieval_config.get("retrieval_weight", 0.5)
                hidden_states = hidden_states + gate * attended
                
                aux_info["retrieval_applied"] = True
        
        return hidden_states, aux_info
    
    return augmented_forward
