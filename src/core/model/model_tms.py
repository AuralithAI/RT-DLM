import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

import optax

from typing import Optional, Literal, Dict, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.model.model_module_self_attention import SelfAttentionModel
from src.core.model.model_transformer_module import TransformerModel
from src.core.ethics.reward_model import EthicalRewardModel
from src.core.model.sparse_moe import SparseMoE
from src.core.model.memory_bank import MemoryBank

# Import reusable components for shared spiking/pruning utilities
from src.core.components.reusable_components import (
    SpikingMechanism,
    PruningManager,
    apply_shared_spiking,
    AttentionConfig,
)

# Import AGI-scale attention mechanisms
from src.core.model.agi_attention import (
    AGIAttention,
    AGIAttentionConfig,
    HierarchicalMemoryFusion,
)

class TMSModel(hk.Module):
    """
    Transformer + MoE + Self-Attention (TMS) Model
    
    This model integrates self-attention, transformer layers, and mixture-of-experts
    with shared spiking/pruning mechanisms from src.core.components.reusable_components.
    
    The SelfAttentionModel now supports advanced attention variants:
    - "standard": Traditional Multi-Head Attention with RoPE
    - "gqa": Grouped-Query Attention (2-4x faster inference)
    - "mqa": Multi-Query Attention (maximum efficiency)
    - "sliding": Sliding Window Attention (for very long sequences)
    - "linear": Linear Attention (O(n) complexity)
    
    The SpikingMechanism and PruningManager from reusable_components provide
    centralized utilities that can be used to:
    - Apply consistent spiking thresholds across all attention layers
    - Track and prune underutilized attention heads and FFN neurons
    - Compute compression ratios for model efficiency analysis
    
    AGI Attention Features:
    - Ring Attention: Distributed attention for infinite context across devices
    - Cross-Memory Attention: Deep interaction between LTM/STM/MTM memory banks
    - Hierarchical Memory Fusion: Attention-based memory consolidation
    
    Example usage with shared spiking:
        # Use shared spiking utility for custom attention scores
        spiked_scores = apply_shared_spiking(scores, spike_threshold=0.1)
        
        # Or use SpikingMechanism for more control
        spiking = SpikingMechanism(spike_threshold=0.1, epsilon=1e-8)
        spiked = spiking.apply(attention_scores)
    
    Example usage with AGI attention:
        # Enable cross-memory attention for memory bank interaction
        model = TMSModel(..., use_agi_attention=True, enable_memory_cross_attention=True)
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        num_layers: int, 
        vocab_size: int, 
        max_seq_length: int, 
        moe_experts: int, 
        moe_top_k: int, 
        memory_size: int, 
        retrieval_k: int, 
        ltm_weight: float, 
        stm_weight: float, 
        mtm_weight: float,
        # Advanced attention configuration
        attention_type: Literal["standard", "gqa", "mqa", "sliding", "linear"] = "standard",
        num_kv_heads: Optional[int] = None,
        position_encoding: Literal["rope", "learned", "none"] = "rope",
        sliding_window_size: int = 512,
        # AGI-scale attention features
        use_agi_attention: bool = False,
        enable_ring_attention: bool = True,
        enable_memory_cross_attention: bool = True,
        enable_infinite_context: bool = False,
        ring_block_size: int = 512,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.use_agi_attention = use_agi_attention
        self.embedding = hk.Embed(vocab_size, d_model)
        self.position_enc = hk.Embed(max_seq_length, d_model)
        
        # Use unified SelfAttentionModel with advanced features
        self.self_attention = SelfAttentionModel(
            d_model=d_model, 
            num_heads=num_heads, 
            vocab_size=vocab_size, 
            max_seq_length=max_seq_length,
            attention_type=attention_type,
            num_kv_heads=num_kv_heads,
            position_encoding=position_encoding,
            sliding_window_size=sliding_window_size
        )
        self.transformer = TransformerModel(d_model, num_heads, num_layers, vocab_size, max_seq_length)
        self.moe = SparseMoE(d_model, moe_experts, moe_top_k, expert_capacity=3)
        self.memory = MemoryBank(memory_size, d_model, retrieval_k)
        self.memory_to_logits = hk.Linear(vocab_size)
        self.memory_projection_ltm = hk.Linear(d_model)
        self.memory_projection_stm = hk.Linear(d_model)
        self.memory_projection_mtm = hk.Linear(d_model)
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.proj = hk.Linear(vocab_size)
        self.ltm_weight = ltm_weight  
        self.stm_weight = stm_weight
        self.mtm_weight = mtm_weight
        self.reward_model = EthicalRewardModel(d_model, vocab_size, max_seq_length * 2)
        self.ethics_weight = hk.get_parameter("ethics_weight", [], init=jnp.ones) * 0.1
        
        # Shared spiking mechanism for consistent behavior across layers
        self._spiking = SpikingMechanism(spike_threshold=0.1, epsilon=1e-8)
        
        # Pruning manager for attention head analysis
        self._pruning_manager = PruningManager(
            num_heads=num_heads,
            d_model=d_model,
            head_threshold=0.01,
            ffn_threshold=0.01
        )
        
        # AGI-scale attention mechanisms
        if use_agi_attention:
            self.agi_attention_config = AGIAttentionConfig(
                d_model=d_model,
                num_heads=num_heads,
                max_seq_length=max_seq_length,
                enable_ring_attention=enable_ring_attention,
                ring_block_size=ring_block_size,
                enable_memory_cross_attention=enable_memory_cross_attention,
                enable_infinite_context=enable_infinite_context,
            )
            self.agi_attention = AGIAttention(self.agi_attention_config)
            
            # Cross-memory attention for LTM/STM/MTM interaction
            if enable_memory_cross_attention:
                self.memory_fusion = HierarchicalMemoryFusion(
                    d_model=d_model,
                    num_heads=num_heads // 2,  # Use fewer heads for memory fusion
                    dropout_rate=0.1
                )
    
    def get_attention_config(self) -> AttentionConfig:
        """Get attention configuration for this model."""
        return AttentionConfig(
            d_model=self.d_model,
            num_heads=self.num_heads,
            spike_threshold=self._spiking.spike_threshold,
            epsilon=self._spiking.epsilon
        )
    
    def apply_shared_spiking(self, scores: jnp.ndarray, spike_threshold: Optional[float] = None) -> jnp.ndarray:
        """
        Apply shared spiking mechanism to attention scores.
        
        This uses the centralized SpikingMechanism from reusable_components
        to ensure consistent spiking behavior across all model layers.
        
        Args:
            scores: Attention scores to apply spiking to
            spike_threshold: Optional override for spike threshold
            
        Returns:
            Spiked attention scores
        """
        if spike_threshold is not None:
            self._spiking.update_threshold(spike_threshold)
        return self._spiking.apply(scores)
    
    def analyze_head_usage(self, attn_weights: jnp.ndarray) -> dict:
        """
        Analyze attention head usage for pruning decisions.
        
        Uses the centralized PruningManager to compute head usage
        statistics and pruning recommendations.
        
        Args:
            attn_weights: Attention weights from forward pass
            
        Returns:
            Dictionary with usage statistics and pruning masks
        """
        head_usage = self._pruning_manager.compute_head_usage(attn_weights)
        ffn_usage = jnp.ones(self.d_model) * 0.1  # Placeholder, would use actual FFN output
        active_heads, active_ffn = self._pruning_manager.get_pruning_mask(head_usage, ffn_usage)
        compression = self._pruning_manager.compute_compression_ratio(active_heads, active_ffn)
        return {
            'head_usage': head_usage,
            'active_heads': active_heads,
            'compression': compression
        }

    def __call__(
        self, 
        inputs, 
        rng=None, 
        return_attention=False, 
        retrieved_memory_ltm=None, 
        retrieved_memory_stm=None, 
        retrieved_memory_mtm=None, 
        spike_threshold=None, 
        epsilon=None, 
        outputs=None, 
        feedback_score=None,
        use_memory_cross_attention: bool = True,
        use_infinite_context: bool = False,
        is_training: bool = True
    ) -> Any:
        """
        Forward pass through the TMS model.
        
        Args:
            inputs: Input token IDs [batch, seq_len]
            rng: Optional random key
            return_attention: Whether to return attention weights
            retrieved_memory_ltm: Long-term memory [batch, d_model]
            retrieved_memory_stm: Short-term memory [batch, d_model]
            retrieved_memory_mtm: Meta-task memory [batch, d_model]
            spike_threshold: Threshold for spiking attention
            epsilon: Small constant for numerical stability
            outputs: Optional outputs for ethical scoring
            feedback_score: Optional feedback for reward model
            use_memory_cross_attention: Use cross-attention for memory fusion (AGI mode)
            use_infinite_context: Use infinite context mode for very long sequences
            is_training: Whether in training mode
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            If return_attention: Also returns attention weights and auxiliary loss
        """
        inputs = jnp.asarray(inputs, dtype=jnp.int32, copy=True)
        x = self.embedding(inputs) + self.position_enc(jnp.arange(inputs.shape[1], dtype=jnp.int32))

        # ** dummy_memory is used to initialize the memory projection layers [Not to be used anywhere else] ** 
        dummy_memory = jnp.zeros((inputs.shape[0], 1, self.embedding.embed_dim), dtype=jnp.float32)
        
        # Memory fusion info for returning
        memory_fusion_info = None
    
        # Check if we should use AGI attention with cross-memory fusion
        all_memories_available = (
            retrieved_memory_ltm is not None and 
            retrieved_memory_stm is not None and 
            retrieved_memory_mtm is not None
        )
        
        if self.use_agi_attention and use_memory_cross_attention and all_memories_available:
            # ==================================================================
            # AGI MODE: Cross-Memory Attention for deep memory interaction
            # Instead of simple weighted sums, memory banks interact via attention
            # ==================================================================
            
            # Project memories to model dimension
            ltm_projected = self.memory_projection_ltm(
                jnp.expand_dims(retrieved_memory_ltm, axis=1)
            ).squeeze(1)  # [batch, d_model]
            stm_projected = self.memory_projection_stm(
                jnp.expand_dims(retrieved_memory_stm, axis=1)
            ).squeeze(1)  # [batch, d_model]
            mtm_projected = self.memory_projection_mtm(
                jnp.expand_dims(retrieved_memory_mtm, axis=1)
            ).squeeze(1)  # [batch, d_model]
            
            # Apply hierarchical memory fusion with cross-attention
            # This allows LTM/STM/MTM to interact and inform each other
            memory_fusion_result = self.memory_fusion(
                ltm=ltm_projected,
                stm=stm_projected,
                mtm=mtm_projected,
                context=x.mean(axis=1),  # Use input as context for importance weighting
                is_training=is_training
            )
            
            # Get fused memory representation
            fused_memory = memory_fusion_result["fused_memory"]  # [batch, d_model]
            memory_fusion_info = {
                "importance_weights": memory_fusion_result.get("importance_weights"),
                "cross_attention_weights": memory_fusion_result.get("cross_attention_weights"),
                "ltm_updated": memory_fusion_result.get("ltm_updated"),
                "stm_updated": memory_fusion_result.get("stm_updated"),
                "mtm_updated": memory_fusion_result.get("mtm_updated"),
            }
            
            # Add fused memory to input representation
            # Expand to sequence dimension and add with learned gating
            fused_memory_expanded = fused_memory[:, None, :].repeat(x.shape[1], axis=1)
            
            # Learn a gate to control memory influence
            memory_gate = hk.Sequential([
                hk.Linear(self.d_model),
                jax.nn.sigmoid
            ], name="memory_gate")
            gate = memory_gate(x)
            x = x + gate * fused_memory_expanded
            
            # Also compute memory logits for output enhancement
            memory_logits = self.memory_to_logits(fused_memory_expanded)
            
        else:
            # ==================================================================
            # LEGACY MODE: Simple weighted sum of memories (original behavior)
            # ==================================================================
            
            # Handle LTM
            if retrieved_memory_ltm is not None:
                retrieved_memory_ltm = self.memory_projection_ltm(
                    jnp.repeat(jnp.expand_dims(retrieved_memory_ltm, axis=1), x.shape[1], axis=1)
                )
                x += self.ltm_weight * retrieved_memory_ltm
                memory_logits = self.memory_to_logits(retrieved_memory_ltm)
            else:
                _ = self.memory_projection_ltm(dummy_memory)
                _ = self.memory_to_logits(dummy_memory)
                memory_logits = None

            # Handle STM
            if retrieved_memory_stm is not None:
                retrieved_memory_stm = self.memory_projection_stm(
                    jnp.repeat(jnp.expand_dims(retrieved_memory_stm, axis=1), x.shape[1], axis=1)
                )
                x += self.stm_weight * retrieved_memory_stm 
            else:
                _ = self.memory_projection_stm(dummy_memory)

            # Handle MTM
            if retrieved_memory_mtm is not None:
                retrieved_memory_mtm = self.memory_projection_mtm(
                    jnp.repeat(jnp.expand_dims(retrieved_memory_mtm, axis=1), x.shape[1], axis=1)
                )
                x += self.mtm_weight * retrieved_memory_mtm
            else:
                _ = self.memory_projection_mtm(dummy_memory)

        # ==================================================================
        # Apply AGI attention (Ring Attention or Infinite Context)
        # ==================================================================
        if self.use_agi_attention and (use_infinite_context or self.agi_attention_config.enable_ring_attention):
            # Use AGI attention for main sequence processing
            x_agi, agi_info = self.agi_attention.forward_attention(
                x, 
                mask=(inputs != 0).astype(jnp.float32),
                is_training=is_training,
                use_infinite_context=use_infinite_context
            )
            x = x + x_agi  # Residual connection
            
            if memory_fusion_info is None:
                memory_fusion_info = {}
            memory_fusion_info["agi_attention_info"] = agi_info

        # Self-attention returns hidden states (not logits) for pipeline
        x, attn_weights_self = self.self_attention(
            inputs, 
            return_attention=True, 
            spike_threshold=spike_threshold, 
            epsilon=epsilon,
            return_hidden_states=True  # Get hidden states for transformer pipeline
        )
        x, attn_weights_transformer = self.transformer(x, rng, return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
        x, top_k_expert_indices, aux_loss, _ = self.moe(x, spike_threshold=spike_threshold, epsilon=epsilon)
        x = self.norm(x)
        logits = self.proj(x)

        if memory_logits is not None:
            logits += memory_logits

        # Compute ethical score
        ethical_loss = 0.0
        ethical_score = None
        if outputs is not None:
            ethical_score = self.reward_model(inputs, outputs)
            if feedback_score is not None:
                ethical_loss = jnp.mean(optax.l2_loss(ethical_score, feedback_score))
                aux_loss += self.ethics_weight * ethical_loss

        if return_attention:
            attention_info = {
                "self_attention": attn_weights_self,
                "transformer": attn_weights_transformer,
                "memory_fusion": memory_fusion_info
            }
            # Return dict with both logits and hidden states for downstream use
            return {
                "logits": logits,
                "hidden_states": x,
                "attention_info": attention_info,
                "top_k_expert_indices": top_k_expert_indices,
                "aux_loss": aux_loss
            }
        return logits
