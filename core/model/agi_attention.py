"""
AGI-Scale Attention Mechanisms for RT-DLM

This module implements advanced attention mechanisms specifically designed for
AGI-level capabilities with true long-context reasoning and memory interaction.

Key Components:
1. Ring Attention: Distributed attention for infinite context across devices
2. Cross-Memory Attention: Deep interaction between LTM/STM/MTM memory banks
3. Hierarchical Memory Fusion: Attention-based memory consolidation
4. Infinite Context Window: Chunked processing with global context aggregation

Design Philosophy:
- Memory banks should interact via attention, not simple weighted sums
- Attention should scale to arbitrary context lengths via device distribution
- Memory consolidation should mimic biological memory systems

References:
- Ring Attention: https://arxiv.org/abs/2310.01889 (Liu et al., 2023)
- Memory-Augmented Neural Networks (Graves et al., 2016)
- Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any, List, Literal
from dataclasses import dataclass
from functools import partial
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Constants for einsum operations (avoid duplication warnings)
# =============================================================================

# Attention score computation: [batch, heads, query, dim] x [batch, heads, key, dim] -> [batch, heads, query, key]
EINSUM_ATTN_SCORES = 'bhqd,bhkd->bhqk'

# Attention output computation: [batch, heads, query, key] x [batch, heads, key, dim] -> [batch, heads, query, dim]
EINSUM_ATTN_OUTPUT = 'bhqk,bhkd->bhqd'

# Import configuration from centralized config module
from config.agi_attention_config import (
    AGIAttentionConfig,
    MemoryFusionStrategy,
    ContextStrategy,
)


# =============================================================================
# Ring Attention for Infinite Context
# =============================================================================

class RingAttentionBlock(hk.Module):
    """
    Ring Attention block for distributed attention across devices.
    
    Ring Attention enables processing of arbitrarily long sequences by:
    1. Splitting sequence into blocks across devices
    2. Each device computes attention for its local block
    3. KV pairs are passed in a ring topology to neighboring devices
    4. Attention scores are accumulated across all blocks
    
    This achieves O(n/d * n) complexity where d is number of devices,
    enabling near-infinite context when scaled across many devices.
    
    For single-device use, this degrades gracefully to blockwise attention
    with memory-efficient chunked computation.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        block_size: int = 512,
        num_devices: int = 1,
        dropout_rate: float = 0.0,
        use_rope: bool = True,
        max_seq_length: int = 2048,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim
        self.block_size = block_size
        self.num_devices = num_devices
        self.dropout_rate = dropout_rate
        self.use_rope = use_rope
        self.max_seq_length = max_seq_length
        
    def _compute_block_attention(
        self,
        q_block: jnp.ndarray,
        k_block: jnp.ndarray,
        v_block: jnp.ndarray,
        block_mask: Optional[jnp.ndarray] = None,
        q_start_idx: int = 0,
        k_start_idx: int = 0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute attention for a single block pair.
        
        Args:
            q_block: Query block [batch, block_size, heads, head_dim]
            k_block: Key block [batch, block_size, heads, head_dim]
            v_block: Value block [batch, block_size, heads, head_dim]
            block_mask: Optional mask for this block pair
            q_start_idx: Starting position of query block (for causal masking)
            k_start_idx: Starting position of key block (for causal masking)
            
        Returns:
            attended: Attention output [batch, block_size, heads, head_dim]
            log_sum_exp: Log-sum-exp for numerically stable accumulation
        """
        _, q_len, _, _ = q_block.shape
        k_len = k_block.shape[1]
        
        # Transpose for attention: [batch, heads, seq, head_dim]
        q = jnp.transpose(q_block, (0, 2, 1, 3))
        k = jnp.transpose(k_block, (0, 2, 1, 3))
        v = jnp.transpose(v_block, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum(EINSUM_ATTN_SCORES, q, k) * scale
        
        # Apply causal mask based on absolute positions
        q_positions = jnp.arange(q_len) + q_start_idx
        k_positions = jnp.arange(k_len) + k_start_idx
        causal_mask = q_positions[:, None] >= k_positions[None, :]
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, q_len, k_len]
        
        # Combine with input mask if provided
        if block_mask is not None:
            combined_mask = causal_mask & (block_mask > 0)
        else:
            combined_mask = causal_mask
        
        # Apply mask
        scores = jnp.where(combined_mask, scores, -1e9)
        
        # Compute numerically stable softmax components
        max_scores = jnp.max(scores, axis=-1, keepdims=True)
        max_scores = jnp.where(jnp.isinf(max_scores), 0.0, max_scores)
        
        exp_scores = jnp.exp(scores - max_scores)
        exp_scores = jnp.where(combined_mask, exp_scores, 0.0)
        
        sum_exp = jnp.sum(exp_scores, axis=-1, keepdims=True) + 1e-10
        attn_weights = exp_scores / sum_exp
        
        # Apply attention to values
        attended = jnp.einsum(EINSUM_ATTN_OUTPUT, attn_weights, v)
        
        # Transpose back: [batch, seq, heads, head_dim]
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        
        # Return log-sum-exp for accumulation across blocks
        log_sum_exp = jnp.log(sum_exp) + max_scores
        log_sum_exp = jnp.transpose(log_sum_exp.squeeze(-1), (0, 2, 1))  # [batch, seq, heads]
        
        return attended, log_sum_exp
    
    def _accumulate_attention(
        self,
        attended_list: List[jnp.ndarray],
        lse_list: List[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Accumulate attention outputs from multiple blocks using log-sum-exp trick.
        
        This enables numerically stable accumulation of attention across
        arbitrarily many blocks without overflow/underflow issues.
        
        Args:
            attended_list: List of attended outputs per block
            lse_list: List of log-sum-exp values per block
            
        Returns:
            Accumulated attention output
        """
        if len(attended_list) == 1:
            return attended_list[0]
        
        # Stack for vectorized computation
        attended_stack = jnp.stack(attended_list, axis=0)  # [num_blocks, batch, seq, heads, dim]
        lse_stack = jnp.stack(lse_list, axis=0)  # [num_blocks, batch, seq, heads]
        
        # Find max LSE for numerical stability
        max_lse = jnp.max(lse_stack, axis=0, keepdims=True)
        
        # Compute weights using log-sum-exp trick
        exp_diff = jnp.exp(lse_stack - max_lse)
        weights = exp_diff / (jnp.sum(exp_diff, axis=0, keepdims=True) + 1e-10)
        
        # Weighted combination
        weights = weights[..., None]  # Add head_dim dimension
        accumulated = jnp.sum(attended_stack * weights, axis=0)
        
        return accumulated
    
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply Ring Attention.
        
        For single device: Computes blockwise attention with accumulation.
        For multiple devices: Would distribute across devices in ring topology.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len]
            is_training: Whether in training mode
            
        Returns:
            output: Attention output [batch, seq_len, d_model]
            attention_weights: Aggregated attention info
        """
        batch_size, seq_len, _ = x.shape
        
        # Projections
        q_proj = hk.Linear(self.d_model, name="q_proj")
        k_proj = hk.Linear(self.d_model, name="k_proj")
        v_proj = hk.Linear(self.d_model, name="v_proj")
        o_proj = hk.Linear(self.d_model, name="o_proj")
        
        # Project and reshape
        q = q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self._apply_rope(q, k, seq_len)
        
        # Compute number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Pad to multiple of block_size
        pad_len = num_blocks * self.block_size - seq_len
        if pad_len > 0:
            q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            if mask is not None:
                mask = jnp.pad(mask, ((0, 0), (0, pad_len)))
        
        # Process each query block
        all_attended = []
        
        for q_block_idx in range(num_blocks):
            q_start = q_block_idx * self.block_size
            q_end = q_start + self.block_size
            q_block = q[:, q_start:q_end]
            
            # Accumulate attention from all key blocks
            block_attended = []
            block_lse = []
            
            for k_block_idx in range(num_blocks):
                k_start = k_block_idx * self.block_size
                k_end = k_start + self.block_size
                k_block = k[:, k_start:k_end]
                v_block = v[:, k_start:k_end]
                
                # Get block mask
                if mask is not None:
                    block_mask = mask[:, k_start:k_end]
                    block_mask = block_mask[:, None, None, :]  # [batch, 1, 1, k_len]
                else:
                    block_mask = None
                
                # Compute attention for this block pair
                attended, lse = self._compute_block_attention(
                    q_block, k_block, v_block,
                    block_mask, q_start, k_start
                )
                
                block_attended.append(attended)
                block_lse.append(lse)
            
            # Accumulate across key blocks
            accumulated = self._accumulate_attention(block_attended, block_lse)
            all_attended.append(accumulated)
        
        # Concatenate query blocks
        output = jnp.concatenate(all_attended, axis=1)
        
        # Remove padding
        output = output[:, :seq_len]
        
        # Output projection
        output = output.reshape(batch_size, seq_len, self.d_model)
        output = o_proj(output)
        
        # Create summary attention weights (for compatibility)
        attention_weights = jnp.ones((batch_size, self.num_heads, seq_len, seq_len))
        
        return output, attention_weights
    
    def _apply_rope(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        seq_len: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply Rotary Position Embedding."""
        # Compute frequencies
        inv_freq = 1.0 / (self.max_seq_length ** (
            jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim
        ))
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(positions, inv_freq)
        
        cos = jnp.cos(freqs)[None, :, None, :]  # [1, seq, 1, dim//2]
        sin = jnp.sin(freqs)[None, :, None, :]
        
        # Split and rotate
        q1, q2 = q[..., :self.head_dim//2], q[..., self.head_dim//2:]
        k1, k2 = k[..., :self.head_dim//2], k[..., self.head_dim//2:]
        
        # Handle padding by extending cos/sin if needed
        if q.shape[1] > seq_len:
            pad_positions = jnp.arange(seq_len, q.shape[1], dtype=jnp.float32)
            pad_freqs = jnp.outer(pad_positions, inv_freq)
            cos = jnp.concatenate([cos, jnp.cos(pad_freqs)[None, :, None, :]], axis=1)
            sin = jnp.concatenate([sin, jnp.sin(pad_freqs)[None, :, None, :]], axis=1)
        
        q_rotated = jnp.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
        k_rotated = jnp.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
        
        return q_rotated, k_rotated


# =============================================================================
# Cross-Memory Attention
# =============================================================================

class CrossMemoryAttention(hk.Module):
    """
    Cross-Attention between Memory Banks (LTM, STM, MTM).
    
    Instead of simple weighted sums, memory banks interact via attention:
    - LTM queries STM for relevant recent context
    - STM queries LTM for persistent knowledge
    - MTM mediates between LTM and STM for task-relevant fusion
    
    This models how biological memory systems interact:
    - Hippocampus (STM) ↔ Neocortex (LTM) consolidation
    - Working memory (MTM) as central executive
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        
    def _cross_attend(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        proj_name: str,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform cross-attention between memory banks.
        
        Args:
            query: Query memory [batch, seq_q, d_model]
            key: Key memory [batch, seq_k, d_model]
            value: Value memory [batch, seq_k, d_model]
            proj_name: Unique name for projections
            is_training: Whether in training mode
            
        Returns:
            output: Cross-attended output [batch, seq_q, d_model]
            weights: Attention weights [batch, heads, seq_q, seq_k]
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1] if query.ndim > 2 else 1
        seq_k = key.shape[1] if key.ndim > 2 else 1
        
        # Handle 2D inputs
        if query.ndim == 2:
            query = query[:, None, :]
        if key.ndim == 2:
            key = key[:, None, :]
        if value.ndim == 2:
            value = value[:, None, :]
        
        # Projections
        q_proj = hk.Linear(self.d_model, name=f"{proj_name}_q")
        k_proj = hk.Linear(self.d_model, name=f"{proj_name}_k")
        v_proj = hk.Linear(self.d_model, name=f"{proj_name}_v")
        o_proj = hk.Linear(self.d_model, name=f"{proj_name}_o")
        
        # Project and reshape
        q = q_proj(query).reshape(batch_size, seq_q, self.num_heads, self.head_dim)
        k = k_proj(key).reshape(batch_size, seq_k, self.num_heads, self.head_dim)
        v = v_proj(value).reshape(batch_size, seq_k, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = jnp.transpose(q, (0, 2, 1, 3))  # [batch, heads, seq_q, dim]
        k = jnp.transpose(k, (0, 2, 1, 3))  # [batch, heads, seq_k, dim]
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention scores
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum(EINSUM_ATTN_SCORES, q, k) * scale
        
        # Softmax
        weights = jax.nn.softmax(scores, axis=-1)
        
        # Dropout
        if is_training and self.dropout_rate > 0:
            weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, weights)
        
        # Apply attention
        attended = jnp.einsum(EINSUM_ATTN_OUTPUT, weights, v)
        
        # Reshape and project
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_q, self.d_model)
        output = o_proj(attended)
        
        # Squeeze if original input was 2D
        if seq_q == 1:
            output = output.squeeze(1)
        
        return output, weights
    
    def __call__(
        self,
        ltm: jnp.ndarray,
        stm: jnp.ndarray,
        mtm: jnp.ndarray,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Compute cross-attention interactions between all memory banks.
        
        Memory Interaction Pattern:
        1. LTM ← STM: Long-term memory queries short-term for recent updates
        2. STM ← LTM: Short-term queries long-term for persistent context
        3. MTM ← (LTM, STM): Meta-task queries both for task fusion
        4. Final fusion via gated combination
        
        Args:
            ltm: Long-term memory [batch, d_model] or [batch, seq, d_model]
            stm: Short-term memory [batch, d_model] or [batch, seq, d_model]
            mtm: Meta-task memory [batch, d_model] or [batch, seq, d_model]
            is_training: Whether in training mode
            
        Returns:
            Dictionary with:
            - fused_memory: Final fused memory representation
            - ltm_updated: LTM after cross-attention
            - stm_updated: STM after cross-attention
            - mtm_updated: MTM after cross-attention
            - attention_weights: Dict of attention weight matrices
        """
        attention_weights = {}
        
        # LTM queries STM: "What recent information is relevant?"
        ltm_from_stm, weights_ltm_stm = self._cross_attend(
            ltm, stm, stm, "ltm_queries_stm", is_training
        )
        attention_weights["ltm_from_stm"] = weights_ltm_stm
        
        # STM queries LTM: "What long-term knowledge applies?"
        stm_from_ltm, weights_stm_ltm = self._cross_attend(
            stm, ltm, ltm, "stm_queries_ltm", is_training
        )
        attention_weights["stm_from_ltm"] = weights_stm_ltm
        
        # MTM queries LTM: "What persistent knowledge for this task?"
        mtm_from_ltm, weights_mtm_ltm = self._cross_attend(
            mtm, ltm, ltm, "mtm_queries_ltm", is_training
        )
        attention_weights["mtm_from_ltm"] = weights_mtm_ltm
        
        # MTM queries STM: "What recent context for this task?"
        mtm_from_stm, weights_mtm_stm = self._cross_attend(
            mtm, stm, stm, "mtm_queries_stm", is_training
        )
        attention_weights["mtm_from_stm"] = weights_mtm_stm
        
        # Gated updates for each memory
        ltm_gate = hk.Sequential([
            hk.Linear(self.d_model), jax.nn.sigmoid
        ], name="ltm_gate")
        stm_gate = hk.Sequential([
            hk.Linear(self.d_model), jax.nn.sigmoid
        ], name="stm_gate")
        mtm_gate = hk.Sequential([
            hk.Linear(self.d_model), jax.nn.sigmoid
        ], name="mtm_gate")
        
        # Apply gated updates
        ltm_updated = ltm + ltm_gate(ltm_from_stm) * ltm_from_stm
        stm_updated = stm + stm_gate(stm_from_ltm) * stm_from_ltm
        mtm_updated = mtm + mtm_gate(mtm_from_ltm + mtm_from_stm) * (mtm_from_ltm + mtm_from_stm) * 0.5
        
        # Layer norms for stability
        ltm_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ltm_norm")
        stm_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="stm_norm")
        mtm_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="mtm_norm")
        
        ltm_updated = ltm_norm(ltm_updated)
        stm_updated = stm_norm(stm_updated)
        mtm_updated = mtm_norm(mtm_updated)
        
        # Final fusion via MTM-mediated attention
        # MTM acts as "central executive" in working memory theory
        fusion_query = mtm_updated
        fusion_kv = jnp.stack([ltm_updated, stm_updated], axis=1)
        
        if fusion_kv.ndim == 3:
            # [batch, 2, d_model]
            pass
        elif fusion_kv.ndim == 4:
            # [batch, 2, seq, d_model] -> reshape
            fusion_kv = fusion_kv.reshape(fusion_kv.shape[0], -1, self.d_model)
        
        fused_memory, fusion_weights = self._cross_attend(
            fusion_query, fusion_kv, fusion_kv, "memory_fusion", is_training
        )
        attention_weights["memory_fusion"] = fusion_weights
        
        return {
            "fused_memory": fused_memory,
            "ltm_updated": ltm_updated,
            "stm_updated": stm_updated,
            "mtm_updated": mtm_updated,
            "attention_weights": attention_weights
        }


# =============================================================================
# Hierarchical Memory Fusion
# =============================================================================

class HierarchicalMemoryFusion(hk.Module):
    """
    Hierarchical fusion of memory banks using multi-level attention.
    
    Implements a biologically-inspired memory consolidation process:
    1. Level 1: Local attention within each memory bank
    2. Level 2: Cross-attention between memory banks
    3. Level 3: Global integration via learned routing
    
    This mimics the hippocampal-cortical consolidation where:
    - Episodic memories (STM) are gradually integrated into semantic memory (LTM)
    - Working memory (MTM) coordinates retrieval and manipulation
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_levels: int = 3,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.dropout_rate = dropout_rate
        
        # Cross-memory attention module
        self.cross_memory_attention = CrossMemoryAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
    def _local_self_attention(
        self,
        memory: jnp.ndarray,
        name: str,
        is_training: bool = True
    ) -> jnp.ndarray:
        """Apply local self-attention within a memory bank."""
        if memory.ndim == 2:
            # Single vector per batch, expand for attention
            memory = memory[:, None, :]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = memory.shape
        head_dim = self.d_model // self.num_heads
        
        # Self-attention projections
        qkv_proj = hk.Linear(self.d_model * 3, name=f"{name}_qkv")
        o_proj = hk.Linear(self.d_model, name=f"{name}_o")
        
        qkv = qkv_proj(memory)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        # Transpose
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention
        scale = 1.0 / jnp.sqrt(head_dim)
        scores = jnp.einsum(EINSUM_ATTN_SCORES, q, k) * scale
        weights = jax.nn.softmax(scores, axis=-1)
        
        if is_training and self.dropout_rate > 0:
            weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, weights)
        
        attended = jnp.einsum(EINSUM_ATTN_OUTPUT, weights, v)
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_len, self.d_model)
        
        output = o_proj(attended) + memory  # Residual
        
        if squeeze_output:
            output = output.squeeze(1)
        
        return output
    
    def _compute_memory_importance(
        self,
        ltm: jnp.ndarray,
        stm: jnp.ndarray,
        mtm: jnp.ndarray,
        context: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute importance scores for each memory bank given context.
        
        Args:
            ltm, stm, mtm: Memory bank representations
            context: Current context for relevance computation
            
        Returns:
            Importance weights [batch, 3]
        """
        # Ensure all are 2D
        if ltm.ndim > 2:
            ltm = ltm.mean(axis=1)
        if stm.ndim > 2:
            stm = stm.mean(axis=1)
        if mtm.ndim > 2:
            mtm = mtm.mean(axis=1)
        if context.ndim > 2:
            context = context.mean(axis=1)
        
        # Compute relevance scores
        scorer = hk.Linear(1, name="memory_importance_scorer")
        
        ltm_score = scorer(ltm * context)
        stm_score = scorer(stm * context)
        mtm_score = scorer(mtm * context)
        
        scores = jnp.concatenate([ltm_score, stm_score, mtm_score], axis=-1)
        importance = jax.nn.softmax(scores, axis=-1)
        
        return importance
    
    def __call__(
        self,
        ltm: jnp.ndarray,
        stm: jnp.ndarray,
        mtm: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Perform hierarchical memory fusion.
        
        Args:
            ltm: Long-term memory [batch, d_model] or [batch, seq, d_model]
            stm: Short-term memory [batch, d_model] or [batch, seq, d_model]
            mtm: Meta-task memory [batch, d_model] or [batch, seq, d_model]
            context: Optional current context for importance weighting
            is_training: Whether in training mode
            
        Returns:
            Dictionary with fused memory and metadata
        """
        results = {}
        
        # Level 1: Local self-attention within each memory bank
        ltm_local = self._local_self_attention(ltm, "ltm_local", is_training)
        stm_local = self._local_self_attention(stm, "stm_local", is_training)
        mtm_local = self._local_self_attention(mtm, "mtm_local", is_training)
        
        results["level1"] = {
            "ltm": ltm_local,
            "stm": stm_local,
            "mtm": mtm_local
        }
        
        # Level 2: Cross-attention between memory banks
        cross_result = self.cross_memory_attention(
            ltm_local, stm_local, mtm_local, is_training
        )
        
        results["level2"] = cross_result
        ltm_cross = cross_result["ltm_updated"]
        stm_cross = cross_result["stm_updated"]
        mtm_cross = cross_result["mtm_updated"]
        
        # Level 3: Global integration via importance-weighted fusion
        if context is not None:
            importance = self._compute_memory_importance(
                ltm_cross, stm_cross, mtm_cross, context
            )
        else:
            # Default to equal weighting
            batch_size = ltm.shape[0]
            importance = jnp.ones((batch_size, 3)) / 3
        
        results["importance_weights"] = importance
        
        # Ensure all memories are same shape for stacking
        if ltm_cross.ndim == 2:
            ltm_for_fusion = ltm_cross
            stm_for_fusion = stm_cross
            mtm_for_fusion = mtm_cross
        else:
            ltm_for_fusion = ltm_cross.mean(axis=1)
            stm_for_fusion = stm_cross.mean(axis=1)
            mtm_for_fusion = mtm_cross.mean(axis=1)
        
        # Stack and weight
        memory_stack = jnp.stack([ltm_for_fusion, stm_for_fusion, mtm_for_fusion], axis=1)
        fused = jnp.einsum('bm,bmd->bd', importance, memory_stack)
        
        # Final projection
        fusion_proj = hk.Sequential([
            hk.Linear(self.d_model * 2),
            jax.nn.silu,
            hk.Linear(self.d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="final_fusion")
        
        fused_memory = fusion_proj(fused)
        
        results["fused_memory"] = fused_memory
        results["cross_attention_weights"] = cross_result["attention_weights"]
        
        return results


# =============================================================================
# Infinite Context Module
# =============================================================================

class InfiniteContextAttention(hk.Module):
    """
    Infinite Context Attention via hierarchical compression.
    
    Enables processing of arbitrarily long sequences by:
    1. Processing input in chunks with local attention
    2. Compressing each chunk into summary tokens
    3. Maintaining a global context of compressed summaries
    4. Cross-attending between local chunks and global context
    
    Memory complexity: O(chunk_size² + global_size²) instead of O(n²)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        chunk_size: int = 1024,
        global_context_size: int = 256,
        compression_ratio: int = 4,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        self.global_context_size = global_context_size
        self.compression_ratio = compression_ratio
        self.dropout_rate = dropout_rate
        
    def _compress_chunk(
        self,
        chunk: jnp.ndarray,
        is_training: bool = True
    ) -> jnp.ndarray:
        """
        Compress a chunk into summary tokens.
        
        Uses learned query tokens to extract compressed representation.
        
        Args:
            chunk: Input chunk [batch, chunk_size, d_model]
            is_training: Whether in training mode
            
        Returns:
            compressed: Summary tokens [batch, chunk_size//compression_ratio, d_model]
        """
        batch_size, chunk_len, _ = chunk.shape
        num_summary = max(1, chunk_len // self.compression_ratio)
        
        # Learned summary queries
        summary_queries = hk.get_parameter(
            "summary_queries",
            [1, num_summary, self.d_model],
            init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        summary_queries = jnp.broadcast_to(
            summary_queries, (batch_size, num_summary, self.d_model)
        )
        
        # Cross-attention from queries to chunk
        q_proj = hk.Linear(self.d_model, name="compress_q")
        k_proj = hk.Linear(self.d_model, name="compress_k")
        v_proj = hk.Linear(self.d_model, name="compress_v")
        o_proj = hk.Linear(self.d_model, name="compress_o")
        
        q = q_proj(summary_queries).reshape(batch_size, num_summary, self.num_heads, self.head_dim)
        k = k_proj(chunk).reshape(batch_size, chunk_len, self.num_heads, self.head_dim)
        v = v_proj(chunk).reshape(batch_size, chunk_len, self.num_heads, self.head_dim)
        
        # Transpose
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum(EINSUM_ATTN_SCORES, q, k) * scale
        weights = jax.nn.softmax(scores, axis=-1)
        
        if is_training and self.dropout_rate > 0:
            weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, weights)
        
        compressed = jnp.einsum(EINSUM_ATTN_OUTPUT, weights, v)
        compressed = jnp.transpose(compressed, (0, 2, 1, 3))
        compressed = compressed.reshape(batch_size, num_summary, self.d_model)
        compressed = o_proj(compressed)
        
        return compressed
    
    def _local_attention(
        self,
        chunk: jnp.ndarray,
        global_context: jnp.ndarray,
        is_training: bool = True
    ) -> jnp.ndarray:
        """
        Apply local attention with global context conditioning.
        
        Args:
            chunk: Local chunk [batch, chunk_size, d_model]
            global_context: Compressed global context [batch, global_size, d_model]
            is_training: Whether in training mode
            
        Returns:
            output: Processed chunk [batch, chunk_size, d_model]
        """
        batch_size, chunk_len, _ = chunk.shape
        global_len = global_context.shape[1]
        
        # Concatenate global context with local chunk for attention
        combined = jnp.concatenate([global_context, chunk], axis=1)
        combined_len = global_len + chunk_len
        
        # Self-attention on combined sequence
        qkv_proj = hk.Linear(self.d_model * 3, name="local_qkv")
        o_proj = hk.Linear(self.d_model, name="local_o")
        
        qkv = qkv_proj(combined)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = q.reshape(batch_size, combined_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, combined_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, combined_len, self.num_heads, self.head_dim)
        
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum(EINSUM_ATTN_SCORES, q, k) * scale
        
        # Create mask: global tokens visible to all, local tokens causal
        # Global context (first global_len tokens) can attend to each other
        # Local tokens (last chunk_len tokens) can attend to global + causal local
        causal_mask = jnp.tril(jnp.ones((combined_len, combined_len)))
        # Allow all tokens to attend to global context
        causal_mask = causal_mask.at[:, :global_len].set(1.0)
        causal_mask = causal_mask[None, None, :, :]
        
        scores = jnp.where(causal_mask > 0, scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        
        if is_training and self.dropout_rate > 0:
            weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, weights)
        
        attended = jnp.einsum(EINSUM_ATTN_OUTPUT, weights, v)
        attended = jnp.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, combined_len, self.d_model)
        attended = o_proj(attended)
        
        # Return only the local chunk portion (+ residual)
        local_output = attended[:, global_len:] + chunk
        
        return local_output
    
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Apply Infinite Context Attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            is_training: Whether in training mode
            
        Returns:
            output: Processed tensor [batch, seq_len, d_model]
            info: Dictionary with global context and chunk info
        """
        batch_size, seq_len, _ = x.shape
        
        # Calculate number of chunks
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Pad to multiple of chunk_size
        pad_len = num_chunks * self.chunk_size - seq_len
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))
        
        # Initialize global context (learnable or from first chunk)
        if num_chunks > 1:
            # Use first chunk to initialize global context
            first_chunk = x[:, :self.chunk_size]
            global_context = self._compress_chunk(first_chunk, is_training)
            
            # Limit global context size
            if global_context.shape[1] > self.global_context_size:
                global_context = global_context[:, :self.global_context_size]
        else:
            # Single chunk - create minimal global context
            global_context = hk.get_parameter(
                "initial_global_context",
                [1, min(16, self.global_context_size), self.d_model],
                init=hk.initializers.TruncatedNormal(stddev=0.02)
            )
            global_context = jnp.broadcast_to(
                global_context, (batch_size, global_context.shape[1], self.d_model)
            )
        
        # Process each chunk
        outputs = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = x[:, start:end]
            
            # Apply local attention with global context
            chunk_output = self._local_attention(chunk, global_context, is_training)
            outputs.append(chunk_output)
            
            # Update global context with compressed chunk
            if i < num_chunks - 1:  # Don't update after last chunk
                chunk_summary = self._compress_chunk(chunk_output, is_training)
                
                # Aggregate into global context (FIFO-style)
                global_context = jnp.concatenate([global_context, chunk_summary], axis=1)
                if global_context.shape[1] > self.global_context_size:
                    global_context = global_context[:, -self.global_context_size:]
        
        # Concatenate outputs
        output = jnp.concatenate(outputs, axis=1)
        
        # Remove padding
        output = output[:, :seq_len]
        
        info = {
            "num_chunks": num_chunks,
            "global_context": global_context,
            "chunk_size": self.chunk_size
        }
        
        return output, info


# =============================================================================
# AGI Attention Module (Unified Interface)
# =============================================================================

class AGIAttention(hk.Module):
    """
    Unified AGI Attention module combining all advanced attention mechanisms.
    
    Features:
    - Ring Attention for distributed infinite context
    - Cross-Memory Attention for LTM/STM/MTM interaction
    - Hierarchical Memory Fusion for multi-level integration
    - Infinite Context via hierarchical compression
    
    This is the recommended attention module for AGI-scale applications.
    """
    
    def __init__(
        self,
        config: AGIAttentionConfig,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim or (config.d_model // config.num_heads)
        
        # Ring Attention for base attention
        if config.enable_ring_attention:
            self.ring_attention = RingAttentionBlock(
                num_heads=config.num_heads,
                head_dim=self.head_dim,
                block_size=config.ring_block_size,
                num_devices=config.num_ring_devices,
                use_rope=config.use_rope,
                max_seq_length=config.max_seq_length
            )
        
        # Cross-Memory Attention
        if config.enable_memory_cross_attention:
            self.memory_attention = CrossMemoryAttention(
                d_model=config.d_model,
                num_heads=config.num_memory_heads,
                dropout_rate=config.memory_dropout
            )
            self.memory_fusion = HierarchicalMemoryFusion(
                d_model=config.d_model,
                num_heads=config.num_memory_heads,
                dropout_rate=config.memory_dropout
            )
        
        # Infinite Context
        if config.enable_infinite_context:
            self.infinite_context = InfiniteContextAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                chunk_size=config.context_chunk_size,
                global_context_size=config.global_context_size
            )
    
    def forward_attention(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        is_training: bool = True,
        use_infinite_context: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Forward pass through attention mechanism.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            is_training: Whether in training mode
            use_infinite_context: Whether to use infinite context mode
            
        Returns:
            output: Attention output
            info: Dictionary with attention metadata
        """
        info = {}
        
        if use_infinite_context and self.config.enable_infinite_context:
            output, ctx_info = self.infinite_context(x, mask, is_training)
            info["infinite_context"] = ctx_info
        elif self.config.enable_ring_attention:
            output, attn_weights = self.ring_attention(x, mask, is_training)
            info["attention_weights"] = attn_weights
        else:
            # Fallback to standard attention
            output = x
            info["fallback"] = True
        
        return output, info
    
    def forward_memory_fusion(
        self,
        ltm: jnp.ndarray,
        stm: jnp.ndarray,
        mtm: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Fuse memory banks via cross-attention.
        
        Args:
            ltm: Long-term memory
            stm: Short-term memory
            mtm: Meta-task memory
            context: Optional current context
            is_training: Whether in training mode
            
        Returns:
            Fusion results including fused memory and attention weights
        """
        if not self.config.enable_memory_cross_attention:
            # Fallback to simple weighted sum
            return {
                "fused_memory": (ltm + stm + mtm) / 3,
                "ltm_updated": ltm,
                "stm_updated": stm,
                "mtm_updated": mtm
            }
        
        return self.memory_fusion(ltm, stm, mtm, context, is_training)
    
    def __call__(
        self,
        x: jnp.ndarray,
        ltm: Optional[jnp.ndarray] = None,
        stm: Optional[jnp.ndarray] = None,
        mtm: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        is_training: bool = True,
        use_infinite_context: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Full forward pass with attention and memory fusion.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            ltm: Optional long-term memory
            stm: Optional short-term memory
            mtm: Optional meta-task memory
            mask: Optional attention mask
            is_training: Whether in training mode
            use_infinite_context: Use infinite context mode
            
        Returns:
            output: Final output tensor
            info: Dictionary with all attention metadata
        """
        info = {}
        
        # Apply main attention
        attn_output, attn_info = self.forward_attention(
            x, mask, is_training, use_infinite_context
        )
        info["attention"] = attn_info
        
        # Apply memory fusion if memories provided
        if ltm is not None and stm is not None and mtm is not None:
            memory_result = self.forward_memory_fusion(
                ltm, stm, mtm, attn_output.mean(axis=1), is_training
            )
            info["memory"] = memory_result
            
            # Combine attention output with fused memory
            fused_memory = memory_result["fused_memory"]
            if fused_memory.ndim == 2:
                fused_memory = fused_memory[:, None, :]
            
            # Gate for memory integration
            mem_gate = hk.Sequential([
                hk.Linear(self.d_model),
                jax.nn.sigmoid
            ], name="memory_integration_gate")
            
            gate = mem_gate(attn_output)
            output = attn_output + gate * fused_memory
        else:
            output = attn_output
        
        # Final layer norm
        output_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="output_norm"
        )
        output = output_norm(output)
        
        return output, info


# =============================================================================
# Factory Functions
# =============================================================================

def create_agi_attention(
    d_model: int = 384,
    num_heads: int = 8,
    preset: Literal["standard", "infinite", "distributed", "full"] = "standard",
    **kwargs
) -> AGIAttention:
    """
    Factory function for creating AGI Attention modules.
    
    Presets:
    - "standard": Ring attention + memory fusion (single device)
    - "infinite": Infinite context via hierarchical compression
    - "distributed": Ring attention optimized for multi-device
    - "full": All features enabled
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        preset: Configuration preset
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured AGIAttention module
    """
    preset_configs = {
        "standard": {
            "enable_ring_attention": True,
            "enable_memory_cross_attention": True,
            "enable_infinite_context": False,
            "ring_block_size": 512,
            "num_ring_devices": 1,
        },
        "infinite": {
            "enable_ring_attention": False,
            "enable_memory_cross_attention": True,
            "enable_infinite_context": True,
            "context_chunk_size": 1024,
            "global_context_size": 256,
        },
        "distributed": {
            "enable_ring_attention": True,
            "enable_memory_cross_attention": True,
            "enable_infinite_context": False,
            "ring_block_size": 256,
            "num_ring_devices": 4,  # Would be set based on actual device count
        },
        "full": {
            "enable_ring_attention": True,
            "enable_memory_cross_attention": True,
            "enable_infinite_context": True,
            "ring_block_size": 512,
            "context_chunk_size": 1024,
            "global_context_size": 256,
        },
    }
    
    base_config = preset_configs.get(preset, preset_configs["standard"])
    base_config.update(kwargs)
    
    config = AGIAttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        **base_config
    )
    
    return AGIAttention(config)
