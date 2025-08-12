import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

import optax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_attention.model_module_self_attention import SelfAttentionModel
from transformer_block.model_transformer_module import TransformerModel
from ethics.reward_model import EthicalRewardModel
from moe_block.sparse_moe import SparseMoE
from TMS_block.memory_bank import MemoryBank

class TMSModel(hk.Module):
    """
    Transformer + MoE + Self-Attention (TMS) Model
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, 
                 moe_experts: int, moe_top_k: int, memory_size: int, retrieval_k: int, ltm_weight: float, stm_weight: float, mtm_weight: float, name=None):
        super().__init__(name=name)
        self.embedding = hk.Embed(vocab_size, d_model)
        self.position_enc = hk.Embed(max_seq_length, d_model)
        self.self_attention = SelfAttentionModel(d_model, num_heads, vocab_size, max_seq_length)
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

    def __call__(self, inputs, rng=None, return_attention=False, retrieved_memory_ltm=None, retrieved_memory_stm=None, retrieved_memory_mtm=None, spike_threshold=None, epsilon=None, outputs=None, feedback_score=None):
        inputs = jnp.asarray(inputs, dtype=jnp.int32, copy=True)
        x = self.embedding(inputs) + self.position_enc(jnp.arange(inputs.shape[1], dtype=jnp.int32))

        # ** dummy_memory is used to initialize the memory projection layers [Not to be used anywhere else] ** 
        dummy_memory = jnp.zeros((inputs.shape[0], 1, self.embedding.embed_dim), dtype=jnp.float32)
    
        # Handle LTM
        if retrieved_memory_ltm is not None:
            retrieved_memory_ltm = self.memory_projection_ltm(jnp.repeat(jnp.expand_dims(retrieved_memory_ltm, axis=1), x.shape[1], axis=1))
            x += self.ltm_weight * retrieved_memory_ltm
            memory_logits = self.memory_to_logits(retrieved_memory_ltm)
        else:
            _ = self.memory_projection_ltm(dummy_memory)
            _ = self.memory_to_logits(dummy_memory)
            memory_logits = None

        # Handle STM
        if retrieved_memory_stm is not None:
            retrieved_memory_stm = self.memory_projection_stm(jnp.repeat(jnp.expand_dims(retrieved_memory_stm, axis=1), x.shape[1], axis=1))
            x += self.stm_weight * retrieved_memory_stm 
        else:
            _ = self.memory_projection_stm(dummy_memory)

        # Handle MTM
        if retrieved_memory_mtm is not None:
            retrieved_memory_mtm = self.memory_projection_mtm(jnp.repeat(jnp.expand_dims(retrieved_memory_mtm, axis=1), x.shape[1], axis=1))
            x += self.mtm_weight * retrieved_memory_mtm
        else:
            _ = self.memory_projection_mtm(dummy_memory)

        x, attn_weights_self = self.self_attention(inputs, return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
        x, attn_weights_transformer = self.transformer(x, rng, return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
        x, top_k_expert_indices, aux_loss = self.moe(x, spike_threshold=spike_threshold, epsilon=epsilon)
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
            return logits, (attn_weights_self, attn_weights_transformer), top_k_expert_indices, aux_loss
        return logits